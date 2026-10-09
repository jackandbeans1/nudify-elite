"""nudify-elite core. Pluggable backends."""
from __future__ import annotations
import os
import numpy as np
import torch
import cv2
from dataclasses import dataclass
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import (FluxFillPipeline, StableDiffusionControlNetInpaintPipeline, StableDiffusionXLControlNetInpaintPipeline, AutoPipelineForInpainting, AutoPipelineForImage2Image, ControlNetModel, AutoencoderKL)
from controlnet_aux import OpenposeDetector

CLOTHING = {4, 5, 6, 7, 8}
PARSER_ID = "mattmdjaga/segformer_b2_clothes"

# mode: cn_inpaint (mask+pose) | auto_inpaint (mask) | auto (detect) | auto_img2img (masked img2img)
BACKENDS = {
    "sd15": {
        "mode": "cn_inpaint", "label": "SD1.5 + pose (fast)",
        "model": os.getenv("SD15_MODEL", "Lykon/dreamshaper-8-inpainting"),
        "cn":    os.getenv("SD15_CN", "lllyasviel/control_v11p_sd15_openpose"),
        "vae":   os.getenv("SD15_VAE", ""),
        "maxside": int(os.getenv("SD15_MAXSIDE", "768")),
        "steps": 25, "guidance": 6.5,
    },
    "sdxl": {
        "mode": "cn_inpaint", "label": "SDXL + pose (quality)",
        "model": os.getenv("SDXL_MODEL", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"),
        "cn":    os.getenv("SDXL_CN", "xinsir/controlnet-openpose-sdxl-1.0"),
        "vae":   os.getenv("SDXL_VAE", "madebyollin/sdxl-vae-fp16-fix"),
        "maxside": int(os.getenv("SDXL_MAXSIDE", "1024")),
        "steps": 30, "guidance": 6.5,
    },
    "flux": {
        "mode": "auto_inpaint", "label": "Flux Fill (inpaint)",
        "model": os.getenv("FLUX_MODEL", "black-forest-labs/FLUX.1-Fill-dev"),
        "cn": "", "vae": "",
        "maxside": int(os.getenv("FLUX_MAXSIDE", "1024")),
        "steps": 28, "guidance": 3.5,
    },
    "zimage": {
        "mode": "auto", "label": "Z-Image (auto)",
        "model": os.getenv("ZIMAGE_MODEL", ""),
        "cn": "", "vae": "",
        "maxside": int(os.getenv("ZIMAGE_MAXSIDE", "1024")),
        "steps": 28, "guidance": 4.0,
    },
    "qwen": {
        "mode": "auto", "label": "Qwen-Image (auto)",
        "model": os.getenv("QWEN_MODEL", ""),
        "cn": "", "vae": "",
        "maxside": int(os.getenv("QWEN_MAXSIDE", "1024")),
        "steps": 28, "guidance": 4.0,
    },
    "custom": {
        "mode": "auto", "label": "Custom local model (auto)",
        "model": os.getenv("CUSTOM_MODEL", ""),
        "cn": "", "vae": "",
        "maxside": int(os.getenv("CUSTOM_MAXSIDE", "1024")),
        "steps": 28, "guidance": 4.0,
    },
}

@dataclass
class Params:
    prompt: str = "bare skin, natural body, photorealistic, soft light, detailed skin"
    negative: str = "clothes, fabric, lowres, blurry, deformed, bad anatomy, extra limbs, watermark"
    backend: str = "sd15"
    loras: str = ""
    steps: int = 0
    guidance: float = 0.0
    seed: int = -1
    mask_grow: int = 8
    feather: int = 4

def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _load(cls, model, **kw):
    try:
        if os.path.exists(model):
            return cls.from_single_file(model, **kw)
        return cls.from_pretrained(model, **kw)
    except TypeError:
        kw.pop("safety_checker", None)
        if os.path.exists(model):
            return cls.from_single_file(model, **kw)
        return cls.from_pretrained(model, **kw)

def apply_loras(pipe, spec: str) -> None:
    if not spec:
        return
    if not hasattr(pipe, "_loras"):
        pipe._loras = set()
    names, weights = [], []
    for i, item in enumerate(str(spec).split(",")):
        item = item.strip()
        if not item:
            continue
        w = 1.0
        if ":" in item:
            head, tail = item.rsplit(":", 1)
            try:
                w = float(tail); item = head
            except ValueError:
                w = 1.0
        name = f"lora_{i}"
        if name not in pipe._loras:
            pipe.load_lora_weights(item, adapter_name=name)
            pipe._loras.add(name)
        names.append(name); weights.append(w)
    if names:
        pipe.set_adapters(names, adapter_weights=weights)

class NudifyPipeline:
    def __init__(self, backend: str = "sd15", model: str | None = None, device: str | None = None):
        self.backend = backend if backend in BACKENDS else "sd15"
        self.cfg = dict(BACKENDS[self.backend])
        if model:
            self.cfg["model"] = model
        self.mode = self.cfg["mode"]
        self.device = device or pick_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        model = self.cfg["model"]
        if not model:
            raise RuntimeError(f"backend '{self.backend}' has no model set (set the matching env var)")

        self.p_proc = SegformerImageProcessor.from_pretrained(PARSER_ID)
        self.parser = SegformerForSemanticSegmentation.from_pretrained(PARSER_ID).to(self.device)

        if self.mode == "cn_inpaint":
            self.pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            try:
                self.pose = self.pose.to(self.device)
            except Exception:
                pass
            cnet = ControlNetModel.from_pretrained(self.cfg["cn"], torch_dtype=self.dtype)
            kw = dict(controlnet=cnet, torch_dtype=self.dtype, safety_checker=None)
            if self.cfg["vae"]:
                kw["vae"] = AutoencoderKL.from_pretrained(self.cfg["vae"], torch_dtype=self.dtype)
            cls = StableDiffusionXLControlNetInpaintPipeline if self.backend == "sdxl" else StableDiffusionControlNetInpaintPipeline
            self.pipe = _load(cls, model, **kw)
        elif self.mode == "auto_inpaint":
            if self.backend == "flux":
                self.pipe = _load(FluxFillPipeline, model, torch_dtype=self.dtype, safety_checker=None)
            else:
                self.pipe = _load(AutoPipelineForInpainting, model, torch_dtype=self.dtype, safety_checker=None)
        else:
            try:
                self.pipe = _load(AutoPipelineForInpainting, model, torch_dtype=self.dtype, safety_checker=None)
                self.mode = "auto_inpaint"
            except Exception:
                self.pipe = _load(AutoPipelineForImage2Image, model, torch_dtype=self.dtype, safety_checker=None)
                self.mode = "auto_img2img"

        if os.getenv("OFFLOAD", "0") == "1":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        try:
            self.pipe.enable_vae_slicing()
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

    @torch.no_grad()
    def clothing_mask(self, img: Image.Image, p: Params) -> Image.Image:
        inp = self.p_proc(images=img, return_tensors="pt").to(self.device)
        logits = self.parser(**inp).logits
        seg = torch.nn.functional.interpolate(
            logits, size=img.size[::-1], mode="bilinear",
            align_corners=False).argmax(1)[0].cpu().numpy()
        m = np.isin(seg, list(CLOTHING)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.mask_grow, p.mask_grow))
        m = cv2.dilate(m, k)
        m = cv2.GaussianBlur(m, (p.feather * 2 + 1, p.feather * 2 + 1), 0)
        return Image.fromarray(m)

    def _call(self, **kw):
        try:
            return self.pipe(**kw)
        except TypeError:
            kw.pop("negative_prompt", None)
            return self.pipe(**kw)

    @torch.no_grad()
    def run(self, img: Image.Image, p: Params, progress=None) -> Image.Image:
        def tick(v):
            if progress:
                progress(float(v))
        apply_loras(self.pipe, p.loras)
        tick(0.02)
        img = img.convert("RGB")
        w, h = img.size
        s = self.cfg["maxside"] / max(w, h)
        if s < 1:
            w, h = int(w * s), int(h * s)
        w, h = (w // 8) * 8, (h // 8) * 8
        img = img.resize((w, h), Image.LANCZOS)
        tick(0.05)
        mask = self.clothing_mask(img, p)
        tick(0.12)
        total = int(p.steps) or self.cfg["steps"]
        guidance = float(p.guidance) or self.cfg["guidance"]
        g = torch.Generator(device=self.device)
        if p.seed >= 0:
            g.manual_seed(p.seed)

        def cb(pipe, step, timestep, cb_kwargs):
            tick(0.15 + 0.83 * (step + 1) / total)
            return cb_kwargs

        if self.mode == "cn_inpaint":
            pose = self.pose(img)
            tick(0.15)
            res = self._call(prompt=p.prompt, negative_prompt=p.negative,
                             image=img, mask_image=mask, control_image=pose,
                             num_inference_steps=total, guidance_scale=guidance,
                             generator=g, height=h, width=w, callback_on_step_end=cb)
        elif self.mode == "auto_inpaint":
            res = self._call(prompt=p.prompt, negative_prompt=p.negative,
                             image=img, mask_image=mask,
                             num_inference_steps=total, guidance_scale=guidance,
                             generator=g, height=h, width=w, callback_on_step_end=cb)
        else:
            strength = float(os.getenv("IMG2IMG_STRENGTH", "0.65"))
            res = self._call(prompt=p.prompt, negative_prompt=p.negative,
                             image=img, strength=strength,
                             num_inference_steps=total, guidance_scale=guidance,
                             generator=g, callback_on_step_end=cb)
        tick(1.0)
        return Image.composite(res.images[0], img, mask)
