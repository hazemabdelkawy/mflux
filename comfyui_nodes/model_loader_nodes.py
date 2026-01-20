"""
Model Loader Nodes for MFLUX
============================
Nodes for loading various MFLUX models with quantization support.
"""

import torch
from typing import Tuple, Dict, Any


class MfluxFlux1Loader:
    """Load FLUX.1 models (dev, schnell, dev-fill, dev-redux, dev-depth, dev-kontext, krea-dev)"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model_variant": (
                    [
                        "dev",
                        "schnell",
                        "dev-fill",
                        "dev-redux",
                        "dev-depth",
                        "dev-kontext",
                        "krea-dev",
                    ],
                    {"default": "schnell"},
                ),
                "quantize": (
                    ["none", "4", "8"],
                    {"default": "8"},
                ),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "lora_paths": ("STRING", {"default": "", "multiline": True}),
                "lora_scales": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        model_variant: str,
        quantize: str,
        model_path: str = "",
        lora_paths: str = "",
        lora_scales: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.flux import Flux1

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Parse LoRA paths and scales
        lora_path_list = [p.strip() for p in lora_paths.split("\n") if p.strip()] if lora_paths else None
        lora_scale_list = [float(s.strip()) for s in lora_scales.split(",") if s.strip()] if lora_scales else None

        # Load model
        model = Flux1(
            model_name=model_path if model_path else model_variant,
            quantize=quant,
            lora_paths=lora_path_list,
            lora_scales=lora_scale_list,
        )

        return (
            {
                "model": model,
                "model_type": "flux1",
                "variant": model_variant,
            },
        )


class MfluxFlux2KleinLoader:
    """Load FLUX.2 Klein models (4B, 9B)"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model_size": (["4B", "9B"], {"default": "4B"}),
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "lora_paths": ("STRING", {"default": "", "multiline": True}),
                "lora_scales": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        model_size: str,
        quantize: str,
        model_path: str = "",
        lora_paths: str = "",
        lora_scales: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.flux2 import Flux2Klein

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Parse LoRA paths and scales
        lora_path_list = [p.strip() for p in lora_paths.split("\n") if p.strip()] if lora_paths else None
        lora_scale_list = [float(s.strip()) for s in lora_scales.split(",") if s.strip()] if lora_scales else None

        # Determine model name
        if model_path:
            model_name = model_path
        else:
            model_name = f"flux2-klein-{model_size.lower()}"

        # Load model
        model = Flux2Klein(
            model_name=model_name,
            quantize=quant,
            lora_paths=lora_path_list,
            lora_scales=lora_scale_list,
        )

        return (
            {
                "model": model,
                "model_type": "flux2_klein",
                "size": model_size,
            },
        )


class MfluxZImageTurboLoader:
    """Load Z-Image Turbo model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "lora_paths": ("STRING", {"default": "", "multiline": True}),
                "lora_scales": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        quantize: str,
        model_path: str = "",
        lora_paths: str = "",
        lora_scales: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.z_image import ZImageTurbo

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Parse LoRA paths and scales
        lora_path_list = [p.strip() for p in lora_paths.split("\n") if p.strip()] if lora_paths else None
        lora_scale_list = [float(s.strip()) for s in lora_scales.split(",") if s.strip()] if lora_scales else None

        # Load model
        model = ZImageTurbo(
            model_name=model_path if model_path else None,
            quantize=quant,
            lora_paths=lora_path_list,
            lora_scales=lora_scale_list,
        )

        return (
            {
                "model": model,
                "model_type": "z_image_turbo",
            },
        )


class MfluxFiboLoader:
    """Load FIBO model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "lora_paths": ("STRING", {"default": "", "multiline": True}),
                "lora_scales": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        quantize: str,
        model_path: str = "",
        lora_paths: str = "",
        lora_scales: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.fibo import Fibo

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Parse LoRA paths and scales
        lora_path_list = [p.strip() for p in lora_paths.split("\n") if p.strip()] if lora_paths else None
        lora_scale_list = [float(s.strip()) for s in lora_scales.split(",") if s.strip()] if lora_scales else None

        # Load model
        model = Fibo(
            model_name=model_path if model_path else None,
            quantize=quant,
            lora_paths=lora_path_list,
            lora_scales=lora_scale_list,
        )

        return (
            {
                "model": model,
                "model_type": "fibo",
            },
        )


class MfluxQwenImageLoader:
    """Load Qwen Image models (base, edit)"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model_variant": (["qwen-image", "qwen-image-edit"], {"default": "qwen-image"}),
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "lora_paths": ("STRING", {"default": "", "multiline": True}),
                "lora_scales": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        model_variant: str,
        quantize: str,
        model_path: str = "",
        lora_paths: str = "",
        lora_scales: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.qwen_image import QwenImage

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Parse LoRA paths and scales
        lora_path_list = [p.strip() for p in lora_paths.split("\n") if p.strip()] if lora_paths else None
        lora_scale_list = [float(s.strip()) for s in lora_scales.split(",") if s.strip()] if lora_scales else None

        # Load model
        model = QwenImage(
            model_name=model_path if model_path else model_variant,
            quantize=quant,
            lora_paths=lora_path_list,
            lora_scales=lora_scale_list,
        )

        return (
            {
                "model": model,
                "model_type": "qwen_image",
                "variant": model_variant,
            },
        )


class MfluxSeedVR2Loader:
    """Load SeedVR2 upscaling model"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "quantize": (["none", "4", "8"], {"default": "8"}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        quantize: str,
        model_path: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.seedvr2 import SeedVR2

        # Parse quantization
        quant = None if quantize == "none" else int(quantize)

        # Load model
        model = SeedVR2(
            model_name=model_path if model_path else None,
            quantize=quant,
        )

        return (
            {
                "model": model,
                "model_type": "seedvr2",
            },
        )


class MfluxDepthProLoader:
    """Load DepthPro model for depth estimation"""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MFLUX/loaders"

    def load_model(
        self,
        model_path: str = "",
    ) -> Tuple[Dict[str, Any]]:
        from mflux.models.depth_pro import DepthPro

        # Load model
        model = DepthPro(
            model_name=model_path if model_path else None,
        )

        return (
            {
                "model": model,
                "model_type": "depth_pro",
            },
        )
