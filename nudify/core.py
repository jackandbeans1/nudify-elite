"""nudify-elite core: parse -> mask -> pose -> inpaint(ControlNet) -> composite."""
from __future__ import annotations
import numpy as np, torch, cv2
from dataclasses import dataclass
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import (StableDiffusionControlNetInpaintPipeline,
                       ControlNetModel, LCMScheduler)
from controlnet_aux import OpenposeDetector

CLOTHING = {4, 5, 6, 7, 8}
PARSER_ID = "mattmdjaga/segformer_b2_clothes"
INPAINT_ID = "runwayml/stable-diffusion-inpainting"
POSE_CN_ID = "lllyasviel/control_v11p_sd15_openpose"

@dataclass
class Params:
    prompt: str = "bare skin, natural body, photorealistic, soft light, high detail"
    negative: str = "clothes, fabric, lowres, deformed, bad anatomy, extra limbs, blurry"
    steps: int = 8
    guidance: float = 1.8
    seed: int = -1
    mask_grow: int = 12
    feather: int = 8

def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

class NudifyPipeline:
    def __init__(self, device: str | None = None):
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

        cnet = ControlNetModel.from_pretrained(POSE_CN_ID, torch_dtype=self.dtype)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            INPAINT_ID, controlnet=cnet, torch_dtype=self.dtype,
            safety_checker=None, requires_safety_checker=False,
        ).to(self.device)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        self.pipe.fuse_lora()
        if self.device == "cuda":
            self.pipe.enable_vae_slicing()
        else:
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
        tick(0.02)
        img = img.convert("RGB")
        w, h = (img.width // 8) * 8, (img.height // 8) * 8
        if (w, h) != img.size:
            img = img.resize((w, h), Image.LANCZOS)
        tick(0.05)
        mask = self.clothing_mask(img, p)
        tick(0.10)
        pose = self.pose(img)
        tick(0.15)
        total = max(1, int(p.steps))

        def cb(pipe, step, timestep, cb_kwargs):
            tick(0.15 + 0.83 * (step + 1) / total)
            return cb_kwargs

        g = torch.Generator(device=self.device)
        if p.seed >= 0:
            g.manual_seed(p.seed)
        out = self.pipe(
            prompt=p.prompt, negative_prompt=p.negative,
            image=img, mask_image=mask, control_image=pose,
            num_inference_steps=total, guidance_scale=p.guidance,
            generator=g, height=h, width=w,
            callback_on_step_end=cb,
        ).images[0]
        tick(1.0)
        return Image.composite(out, img, mask)

