"""
Depth and Redux Nodes for MFLUX
================================
Nodes for depth conditioning, Redux style reference, and depth extraction.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, List
import tempfile
from pathlib import Path


class MfluxDepthConditioning:
    """Generate images with depth conditioning using FLUX.1 Depth"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "depth_image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "MFLUX/depth_redux"

    def generate(
        self,
        depth_image: torch.Tensor,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        quantize: str,
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        from mflux.models.flux import Flux1

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Load model
        mflux_model = Flux1(
            model_name="dev-depth",
            quantize=quant,
        )

        # Convert ComfyUI image tensor to PIL Image
        image_np = (depth_image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            pil_image.save(tmp_path)

        try:
            # Generate image with depth conditioning
            result_image = mflux_model.generate_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance=guidance,
                depth_image_path=tmp_path,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        # Convert result to torch tensor
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]

        return (result_tensor,)


class MfluxRedux:
    """Generate images with style reference using FLUX.1 Redux"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "reference_image1": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "redux_strength1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "reference_image2": ("IMAGE",),
                "redux_strength2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "reference_image3": ("IMAGE",),
                "redux_strength3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "MFLUX/depth_redux"

    def generate(
        self,
        reference_image1: torch.Tensor,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        redux_strength1: float,
        quantize: str,
        reference_image2: torch.Tensor = None,
        redux_strength2: float = 1.0,
        reference_image3: torch.Tensor = None,
        redux_strength3: float = 1.0,
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        from mflux.models.flux import Flux1

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Load model
        mflux_model = Flux1(
            model_name="dev-redux",
            quantize=quant,
        )

        # Collect reference images and strengths
        redux_images = [reference_image1]
        redux_strengths = [redux_strength1]

        if reference_image2 is not None:
            redux_images.append(reference_image2)
            redux_strengths.append(redux_strength2)

        if reference_image3 is not None:
            redux_images.append(reference_image3)
            redux_strengths.append(redux_strength3)

        # Convert images to PIL and save to temp files
        tmp_paths = []
        try:
            for img_tensor in redux_images:
                image_np = (img_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)

                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp_path = Path(tmp.name)
                pil_image.save(tmp_path)
                tmp_paths.append(tmp_path)

            # Generate image with style reference
            result_image = mflux_model.generate_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance=guidance,
                redux_image_paths=tmp_paths,
                redux_image_strengths=redux_strengths,
            )
        finally:
            # Clean up temp files
            for tmp_path in tmp_paths:
                tmp_path.unlink(missing_ok=True)

        # Convert result to torch tensor
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]

        return (result_tensor,)


class MfluxDepthExtraction:
    """Extract depth map from image using DepthPro"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "extract_depth"
    CATEGORY = "MFLUX/depth_redux"

    def extract_depth(
        self,
        image: torch.Tensor,
        model_path: str = "",
    ) -> Tuple[torch.Tensor]:
        from mflux.models.depth_pro import DepthPro

        # Load model
        mflux_model = DepthPro(
            model_name=model_path if model_path else None,
        )

        # Convert ComfyUI image tensor to PIL Image
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            pil_image.save(tmp_path)

        try:
            # Extract depth
            depth_image = mflux_model.generate_depth(
                image_path=tmp_path,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        # Convert result to torch tensor
        result_np = np.array(depth_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]

        return (result_tensor,)
