"""
Generation Nodes for MFLUX
===========================
Text-to-image and image-to-image generation nodes.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any


class MfluxTextToImage:
    """Generate images from text prompts using any loaded MFLUX model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MFLUX_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "MFLUX/generation"

    def generate(
        self,
        model: Dict[str, Any],
        prompt: str,
        seed: int,
        steps: int,
        width: int,
        height: int,
        guidance: float,
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        # Get the actual model instance
        mflux_model = model["model"]

        # Generate image
        image = mflux_model.generate_image(
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
        )

        # Convert PIL Image to torch tensor for ComfyUI
        # ComfyUI expects images in shape [B, H, W, C] with values in [0, 1]
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        return (image_tensor,)


class MfluxImageToImage:
    """Generate images from image + text using any loaded MFLUX model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MFLUX_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "MFLUX/generation"

    def generate(
        self,
        model: Dict[str, Any],
        image: torch.Tensor,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        strength: float,
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        import tempfile
        from pathlib import Path

        # Get the actual model instance
        mflux_model = model["model"]

        # Convert ComfyUI image tensor to PIL Image
        # ComfyUI provides images in shape [B, H, W, C] with values in [0, 1]
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Save to temporary file for mflux to load
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
                image_path=tmp_path,
                image_strength=strength,
            )
        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

        # Convert PIL Image to torch tensor for ComfyUI
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]

        return (result_tensor,)
