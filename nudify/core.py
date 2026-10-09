"""nudify-elite core. Tiers: sd15 (fast) / sdxl (quality). Env-swappable models."""
from __future__ import annotations
import os
import numpy as np
import torch
import cv2
from dataclasses import dataclass
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import (StableDiffusionControlNetInpaintPipeline,
                       StableDiffusionXLControlNetInpaintPipeline,
                       ControlNetModel, AutoencoderKL)
from controlnet_aux import OpenposeDetector

CLOTHING = {4, 5, 6, 7, 8}
PARSER_ID = "mattmdjaga/segformer_b2_clothes"

TIERS = {
    "sd15": {
        "model":      os.getenv("SD15_MODEL", "Lykon/dreamshaper-8-inpainting"),
        "controlnet": os.getenv("SD15_CN", "lllyasviel/control_v11p_sd15_openpose"),
        "vae":        os.getenv("SD15_VAE", ""),
        "maxside":    int(os.getenv("SD15_MAXSIDE", "768")),
        "steps": 25, "guidance": 6.5,
    },
    "sdxl": {
        "model":      os.getenv("SDXL_MODEL", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"),
        "controlnet": os.getenv("SDXL_CN", "xinsir/controlnet-openpose-sdxl-1.0"),
        "vae":        os.getenv("SDXL_VAE", "madebyollin/sdxl-vae-fp16-fix"),
        "maxside":    int(os.getenv("SDXL_MAXSIDE", "1024")),
        "steps": 30, "guidance": 6.5,
    },
}

@dataclass
class Params:
    prompt: str = "bare skin, natural body, photorealistic, soft light, detailed skin"
    negative: str = "clothes, fabric, lowres, blurry, deformed, bad anatomy, extra limbs, watermark"
    tier: str = "sd15"
    loras: str = ""
    steps: int = 0
    guidance: float = 0.0
    seed: int = -1
    mask_grow: int = 8
    feather: int = 4

def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

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
    def __init__(self, tier: str = "sd15", device: str | None = None):
        self.tier = tier if tier in TIERS else "sd15"
        self.cfg = TIERS[self.tier]
        self.device = device or pick_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        self.p_proc = SegformerImageProcessor.from_pretrained(PARSER_ID)
        self.parser = SegformerForSemanticSegmentation.from_pretrained(PARSER_ID).to(self.device)
        self.pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        try:
            self.pose = self.pose.to(self.device)
        except Exception:
            pass

        cnet = ControlNetModel.from_pretrained(self.cfg["controlnet"], torch_dtype=self.dtype)
        kw = dict(controlnet=cnet, torch_dtype=self.dtype, safety_checker=None)
        if self.cfg["vae"]:
            kw["vae"] = AutoencoderKL.from_pretrained(self.cfg["vae"], torch_dtype=self.dtype)

        cls = StableDiffusionXLControlNetInpaintPipeline if self.tier == "sdxl" else StableDiffusionControlNetInpaintPipeline
        model = self.cfg["model"]
        if os.path.exists(model):
            self.pipe = cls.from_single_file(model, **kw)
        else:
            self.pipe = cls.from_pretrained(model, requires_safety_checker=False, **kw)
        self.pipe.to(self.device)
        self.pipe.enable_vae_slicing()
        self.pipe.enable_attention_slicing()

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
        tick(0.10)
        pose = self.pose(img)
        tick(0.15)
        total = int(p.steps) or self.cfg["steps"]
        guidance = float(p.guidance) or self.cfg["guidance"]

        def cb(pipe, step, timestep, cb_kwargs):
            tick(0.15 + 0.83 * (step + 1) / total)
            return cb_kwargs

        g = torch.Generator(device=self.device)
        if p.seed >= 0:
            g.manual_seed(p.seed)
        out = self.pipe(
            prompt=p.prompt, negative_prompt=p.negative,
            image=img, mask_image=mask, control_image=pose,
            num_inference_steps=total, guidance_scale=guidance,
            generator=g, height=h, width=w,
            callback_on_step_end=cb,
        ).images[0]
        tick(1.0)
        return Image.composite(out, img, mask)
