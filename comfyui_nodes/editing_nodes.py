"""
Image Editing Nodes for MFLUX
==============================
Nodes for image editing with FLUX.2, Qwen, Kontext, and CatVTON.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
import tempfile
from pathlib import Path


class MfluxFlux2Edit:
    """Edit images using FLUX.2 Klein model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MFLUX_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "MFLUX/editing"

    def edit(
        self,
        model: Dict[str, Any],
        image: torch.Tensor,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        # Get the actual model instance
        mflux_model = model["model"]

        # Convert ComfyUI image tensor to PIL Image
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            pil_image.save(tmp_path)

        try:
            # Edit image
            result_image = mflux_model.edit_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance=guidance,
                image_path=tmp_path,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        # Convert result to torch tensor
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]

        return (result_tensor,)


class MfluxQwenEdit:
    """Edit images using Qwen Image Edit model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MFLUX_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "MFLUX/editing"

    def edit(
        self,
        model: Dict[str, Any],
        image: torch.Tensor,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        # Get the actual model instance
        mflux_model = model["model"]

        # Convert ComfyUI image tensor to PIL Image
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            pil_image.save(tmp_path)

        try:
            # Edit image
            result_image = mflux_model.edit_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance=guidance,
                image_path=tmp_path,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        # Convert result to torch tensor
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]

        return (result_tensor,)


class MfluxKontextEdit:
    """In-context editing using FLUX.1 Kontext model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MFLUX_MODEL",),
                "input_image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "context_image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "MFLUX/editing"

    def edit(
        self,
        model: Dict[str, Any],
        input_image: torch.Tensor,
        mask_image: torch.Tensor,
        context_image: torch.Tensor,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        # Get the actual model instance
        mflux_model = model["model"]

        # Convert images to PIL
        def tensor_to_pil(tensor):
            image_np = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(image_np)

        input_pil = tensor_to_pil(input_image)
        mask_pil = tensor_to_pil(mask_image)
        context_pil = tensor_to_pil(context_image)

        # Save to temporary files
        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_context,
        ):
            input_path = Path(tmp_input.name)
            mask_path = Path(tmp_mask.name)
            context_path = Path(tmp_context.name)

            input_pil.save(input_path)
            mask_pil.save(mask_path)
            context_pil.save(context_path)

        try:
            # Edit image with context
            result_image = mflux_model.generate_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance=guidance,
                image_path=input_path,
                masked_image_path=mask_path,
                context_image_path=context_path,
            )
        finally:
            input_path.unlink(missing_ok=True)
            mask_path.unlink(missing_ok=True)
            context_path.unlink(missing_ok=True)

        # Convert result to torch tensor
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]

        return (result_tensor,)


class MfluxCatVTONEdit:
    """Virtual try-on using CatVTON model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MFLUX_MODEL",),
                "person_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "try_on"
    CATEGORY = "MFLUX/editing"

    def try_on(
        self,
        model: Dict[str, Any],
        person_image: torch.Tensor,
        garment_image: torch.Tensor,
        mask_image: torch.Tensor,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
    ) -> Tuple[torch.Tensor]:
        # Get the actual model instance
        mflux_model = model["model"]

        # Convert images to PIL
        def tensor_to_pil(tensor):
            image_np = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(image_np)

        person_pil = tensor_to_pil(person_image)
        garment_pil = tensor_to_pil(garment_image)
        mask_pil = tensor_to_pil(mask_image)

        # Save to temporary files
        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_person,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_garment,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask,
        ):
            person_path = Path(tmp_person.name)
            garment_path = Path(tmp_garment.name)
            mask_path = Path(tmp_mask.name)

            person_pil.save(person_path)
            garment_pil.save(garment_path)
            mask_pil.save(mask_path)

        try:
            # Generate virtual try-on result
            result_image = mflux_model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                guidance=guidance,
                image_path=person_path,
                garment_image_path=garment_path,
                masked_image_path=mask_path,
            )
        finally:
            person_path.unlink(missing_ok=True)
            garment_path.unlink(missing_ok=True)
            mask_path.unlink(missing_ok=True)

        # Convert result to torch tensor
        result_np = np.array(result_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]

        return (result_tensor,)
