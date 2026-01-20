"""
ControlNet Nodes for MFLUX
===========================
Nodes for ControlNet-based generation (Canny, Upscaler).
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
import tempfile
from pathlib import Path


class MfluxControlNetCanny:
    """Generate images using ControlNet Canny edge detection"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "control_image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "controlnet_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "model_variant": (["dev", "schnell"], {"default": "schnell"}),
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "MFLUX/controlnet"

    def generate(
        self,
        control_image: torch.Tensor,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        controlnet_strength: float,
        model_variant: str,
        quantize: str,
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        from mflux.models.flux_controlnet import FluxControlnet

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Load model
        model_name = f"{model_variant}-controlnet-canny"
        mflux_model = FluxControlnet(
            model_name=model_name,
            quantize=quant,
        )

        # Convert ComfyUI image tensor to PIL Image
        image_np = (control_image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            pil_image.save(tmp_path)

        try:
            # Generate image
            result_image = mflux_model.generate_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance=guidance,
                controlnet_image_path=tmp_path,
                controlnet_strength=controlnet_strength,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        # Convert result to torch tensor
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]

        return (result_tensor,)


class MfluxControlNetUpscaler:
    """Upscale images using ControlNet Upscaler"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "controlnet_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "MFLUX/controlnet"

    def upscale(
        self,
        image: torch.Tensor,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        controlnet_strength: float,
        scale_factor: float,
        quantize: str,
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        from mflux.models.flux_controlnet import FluxControlnet

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Load model
        mflux_model = FluxControlnet(
            model_name="dev-controlnet-upscaler",
            quantize=quant,
        )

        # Convert ComfyUI image tensor to PIL Image
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Resize image for upscaling
        new_width = int(pil_image.width * scale_factor)
        new_height = int(pil_image.height * scale_factor)
        # Ensure dimensions are multiples of 16
        new_width = (new_width // 16) * 16
        new_height = (new_height // 16) * 16

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            pil_image.save(tmp_path)

        try:
            # Upscale image
            result_image = mflux_model.generate_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                height=new_height,
                width=new_width,
                guidance=guidance,
                controlnet_image_path=tmp_path,
                controlnet_strength=controlnet_strength,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        # Convert result to torch tensor
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]

        return (result_tensor,)
