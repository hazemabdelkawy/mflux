# MFLUX ComfyUI Nodes

Comprehensive ComfyUI custom nodes for the MFLUX image generation framework. These nodes provide full access to all MFLUX features including multiple model architectures, image editing, ControlNet, upscaling, and more.

## Features

### Supported Models
- **FLUX.1** (dev, schnell, dev-fill, dev-redux, dev-depth, dev-kontext, krea-dev)
- **FLUX.2 Klein** (4B, 9B) - Latest fast models with editing capabilities
- **Z-Image Turbo** - Best all-rounder: fast, small, very good quality
- **FIBO** - JSON-native prompts with structured control
- **Qwen Image** - Large model with strong multilingual support and editing
- **SeedVR2** - Best upscaling model
- **DepthPro** - Fast & accurate depth estimation

### Core Features
- ✅ Text-to-image generation
- ✅ Image-to-image generation
- ✅ Image editing (FLUX.2, Qwen, FIBO)
- ✅ ControlNet (Canny edge detection, Upscaler)
- ✅ Redux (style reference with multiple images)
- ✅ Depth conditioning
- ✅ Fill/Inpainting
- ✅ In-context editing (Kontext)
- ✅ Virtual try-on (CatVTON)
- ✅ Upscaling (SeedVR2, ControlNet)
- ✅ LoRA support (multi-LoRA with scales)
- ✅ Quantization (4-bit, 8-bit for memory optimization)
- ✅ Depth extraction (DepthPro)

## Automatic Model Downloading

**All models automatically download from HuggingFace when first loaded!**

The ComfyUI nodes are designed to automatically download models from HuggingFace Hub when you first use them. No manual downloading required!

### How it works:
1. **First load**: Models download from HuggingFace (can take a while depending on model size)
2. **Subsequent loads**: Models load from local cache (very fast)
3. **Cache location**: Models are cached in `~/.cache/huggingface/hub/` by default

### Model sizes (approximate):
- **FLUX.1** (dev, schnell, fill, redux, depth, kontext, krea): ~24GB (full) / ~12GB (8-bit)
- **FLUX.2 Klein 4B**: ~8GB (full) / ~4GB (8-bit)
- **FLUX.2 Klein 9B**: ~18GB (full) / ~9GB (8-bit)
- **Z-Image Turbo**: ~12GB (full) / ~6GB (8-bit)
- **FIBO**: ~16GB (full) / ~8GB (8-bit)
- **Qwen Image/Edit**: ~40GB (full) / ~20GB (8-bit)
- **SeedVR2**: ~6GB (full) / ~3GB (8-bit)
- **DepthPro**: ~200MB
- **ControlNet models**: Included with base FLUX.1 models
- **CatVTON**: ~2GB (additional transformer)

### Default model sources:

**FLUX.1 Models:**
- **dev**: `black-forest-labs/FLUX.1-dev`
- **schnell**: `black-forest-labs/FLUX.1-schnell`
- **Fill (dev-fill)**: `black-forest-labs/FLUX.1-Fill-dev`
- **Fill CatVTON (dev-fill-catvton)**: `black-forest-labs/FLUX.1-Fill-dev` + `xiaozaa/catvton-flux-beta`
- **Redux (dev-redux)**: `black-forest-labs/FLUX.1-Redux-dev`
- **Depth (dev-depth)**: `black-forest-labs/FLUX.1-Depth-dev`
- **Kontext (dev-kontext)**: `black-forest-labs/FLUX.1-Kontext-dev`
- **Krea (krea-dev)**: `black-forest-labs/FLUX.1-Krea-dev`

**FLUX.2 Models:**
- **Klein 4B**: `black-forest-labs/FLUX.2-klein-4B`
- **Klein 9B**: `black-forest-labs/FLUX.2-klein-9B`

**Other Models:**
- **Z-Image Turbo**: `Tongyi-MAI/Z-Image-Turbo`
- **FIBO**: `briaai/FIBO`
- **Qwen Image**: `Qwen/Qwen-Image`
- **Qwen Image Edit**: `Qwen/Qwen-Image-Edit-2509`
- **SeedVR2**: `numz/SeedVR2_comfyUI`
- **DepthPro**: Downloads from `https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt` (Apple CDN)

**ControlNet Models:**
- **Canny**: `InstantX/FLUX.1-dev-Controlnet-Canny`
- **Upscaler**: `jasperai/Flux.1-dev-Controlnet-Upscaler`

**Virtual Try-On:**
- **CatVTON**: `xiaozaa/catvton-flux-beta` (custom transformer for dev-fill-catvton)

*All models except DepthPro are downloaded from HuggingFace.*

### Custom models:
You can also use custom models by specifying:
- **Local path**: `/path/to/my/model`
- **HuggingFace repo**: `username/model-name`

The nodes will automatically detect the format and load accordingly!

---

## Installation

### Prerequisites
- ComfyUI installed and working
- Python 3.10+
- MFLUX installed (`pip install mflux`)
- Sufficient disk space for model downloads (see sizes above)

### Install Nodes

#### Method 1: Clone into ComfyUI custom_nodes directory
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourusername/mflux.git
cd mflux
pip install -r requirements.txt
```

#### Method 2: Symlink (for development)
```bash
cd /path/to/ComfyUI/custom_nodes
ln -s /path/to/mflux/comfyui_nodes mflux_nodes
```

#### Method 3: Manual copy
```bash
cp -r /path/to/mflux/comfyui_nodes /path/to/ComfyUI/custom_nodes/mflux_nodes
```

### Restart ComfyUI
After installation, restart ComfyUI to load the new nodes.

---

## Quick Start

1. **Install the nodes** (see Installation above)
2. **Open ComfyUI** and create a new workflow
3. **Add a model loader** (e.g., "MFLUX FLUX.2 Klein Loader")
   - Choose model size (4B or 9B)
   - Set quantization to 8 for good balance of speed and quality
   - Leave model_path empty to auto-download from HuggingFace
   - First run will download the model (be patient!)
4. **Add a generation node** (e.g., "MFLUX Text to Image")
   - Connect the model output from the loader
   - Enter your prompt
   - Adjust steps (4-8 for FLUX.2, 20-30 for others)
   - Set guidance (1.0 for FLUX.2, 3.5-4.0 for others)
5. **Add a Save Image node** to save the result
6. **Queue the workflow** and wait for generation!

**First-time users**: The first run will download models from HuggingFace. This can take 10-30 minutes depending on your connection and model size. Subsequent runs will be much faster as models are cached locally.

---

## Node Reference

### Model Loaders

#### MFLUX FLUX.1 Loader
Loads FLUX.1 models with various variants. **Automatically downloads from HuggingFace on first use.**

**Inputs:**
- `model_variant`: Choose from dev, schnell, dev-fill, dev-fill-catvton, dev-redux, dev-depth, dev-kontext, krea-dev
- `quantize`: none, 4, or 8 (default: 8)
- `model_path` (optional): Leave empty to auto-download, or specify custom path/HuggingFace repo
- `lora_paths` (optional): One LoRA path per line (supports HuggingFace repos)
- `lora_scales` (optional): Comma-separated scales (e.g., "1.0, 0.8")

**Outputs:**
- `model`: MFLUX model ready for generation

**Note**: Leave `model_path` empty to use the default HuggingFace repository for the selected variant. The model will download automatically on first use and be cached for future loads.

#### MFLUX FLUX.2 Klein Loader
Loads FLUX.2 Klein models (4B or 9B). **Automatically downloads from HuggingFace on first use.**

**Inputs:**
- `model_size`: 4B or 9B
- `quantize`: none, 4, or 8 (default: 8)
- `model_path` (optional): Leave empty to auto-download, or specify custom path/HuggingFace repo
- `lora_paths` (optional): One LoRA path per line (supports HuggingFace repos)
- `lora_scales` (optional): Comma-separated scales

**Outputs:**
- `model`: MFLUX model ready for generation

**Note**: Default HuggingFace repos: `black-forest-labs/FLUX.2-klein-4B` and `black-forest-labs/FLUX.2-klein-9B`

#### MFLUX Z-Image Turbo Loader
Loads Z-Image Turbo model for fast, high-quality generation. **Auto-downloads from `Tongyi-MAI/Z-Image-Turbo`.**

#### MFLUX FIBO Loader
Loads FIBO model for JSON-native structured prompts. **Auto-downloads from `briaai/FIBO`.**

#### MFLUX Qwen Image Loader
Loads Qwen Image models (base or edit variant). **Auto-downloads from HuggingFace.**

**Inputs:**
- `model_variant`: qwen-image or qwen-image-edit (default HF repos: `Qwen/Qwen-Image` and `Qwen/Qwen-Image-Edit-2509`)
- `quantize`: none, 4, or 8

#### MFLUX SeedVR2 Loader
Loads SeedVR2 upscaling model. **Auto-downloads from `numz/SeedVR2_comfyUI`.**

#### MFLUX DepthPro Loader
Loads DepthPro model for depth estimation. **Auto-downloads from Apple's CDN** (not HuggingFace).

---

### Generation Nodes

#### MFLUX Text to Image
Generates images from text prompts using any loaded model.

**Inputs:**
- `model`: MFLUX model from a loader
- `prompt`: Text description of desired image
- `seed`: Random seed (0-max int)
- `steps`: Number of inference steps (1-100, default: 4)
- `width`: Image width in pixels (256-2048, multiple of 16, default: 1024)
- `height`: Image height in pixels (256-2048, multiple of 16, default: 1024)
- `guidance`: Guidance scale (0.0-20.0, default: 4.0)
- `negative_prompt` (optional): Things to avoid in generation

**Outputs:**
- `image`: Generated image

**Tips:**
- FLUX.2 Klein works best with guidance=1.0
- Z-Image Turbo works well with steps=4-9
- Higher guidance values give more prompt adherence but less creativity

#### MFLUX Image to Image
Generates images from an input image + text prompt.

**Inputs:**
- `model`: MFLUX model from a loader
- `image`: Input image to transform
- `prompt`: Text description of desired changes
- `seed`: Random seed
- `steps`: Number of inference steps
- `guidance`: Guidance scale
- `strength`: How much to change the input (0.0-1.0, default: 0.8)
- `negative_prompt` (optional): Things to avoid

**Outputs:**
- `image`: Transformed image

**Tips:**
- Lower strength (0.3-0.5) keeps more of original image
- Higher strength (0.7-0.9) allows more dramatic changes

---

### Editing Nodes

#### MFLUX FLUX.2 Edit
Edits images using FLUX.2 Klein's native editing capabilities.

**Inputs:**
- `model`: FLUX.2 Klein model
- `image`: Image to edit
- `prompt`: Description of desired edits
- `seed`: Random seed
- `steps`: Inference steps (default: 4)
- `guidance`: Guidance scale (default: 1.0 for FLUX.2)
- `negative_prompt` (optional): Things to avoid

**Outputs:**
- `image`: Edited image

**Tips:**
- FLUX.2 is optimized for guidance=1.0
- Works with very few steps (4-8) for fast editing

#### MFLUX Qwen Edit
Edits images using Qwen Image Edit model.

**Inputs:**
- `model`: Qwen Image Edit model
- `image`: Image to edit
- `prompt`: Description of desired edits
- `seed`: Random seed
- `steps`: Inference steps (default: 20)
- `guidance`: Guidance scale (default: 3.5)

**Outputs:**
- `image`: Edited image

**Tips:**
- Supports multilingual prompts
- Excellent for complex editing instructions

#### MFLUX Kontext Edit
In-context editing using FLUX.1 Kontext model with mask and reference context.

**Inputs:**
- `model`: FLUX.1 Kontext model
- `input_image`: Base image to edit
- `mask_image`: Mask defining edit region (white=edit, black=preserve)
- `context_image`: Reference image for context
- `prompt`: Edit description
- `seed`: Random seed
- `steps`: Inference steps (default: 25)
- `guidance`: Guidance scale (default: 3.5)

**Outputs:**
- `image`: Edited image with context-aware changes

#### MFLUX CatVTON Edit
Virtual try-on using CatVTON model.

**Inputs:**
- `model`: CatVTON model (loaded as dev-fill-catvton)
- `person_image`: Image of person
- `garment_image`: Image of garment to try on
- `mask_image`: Mask of area to replace
- `prompt`: Description
- `seed`: Random seed
- `steps`: Inference steps (default: 25)
- `guidance`: Guidance scale (default: 2.5)

**Outputs:**
- `image`: Person wearing the garment

---

### ControlNet Nodes

#### MFLUX ControlNet Canny
Generates images guided by Canny edge detection.

**Inputs:**
- `control_image`: Input image (edges will be detected automatically)
- `prompt`: Text description
- `seed`: Random seed
- `steps`: Inference steps (default: 25)
- `guidance`: Guidance scale (default: 4.0)
- `controlnet_strength`: How strongly to follow edges (0.0-1.0, default: 0.8)
- `model_variant`: dev or schnell
- `quantize`: none, 4, or 8

**Outputs:**
- `image`: Generated image following edge structure

**Tips:**
- Use line art or sketches as control images
- Lower strength for looser interpretation
- Higher strength for precise edge adherence

#### MFLUX ControlNet Upscaler
Upscales images using ControlNet-guided generation.

**Inputs:**
- `image`: Image to upscale
- `prompt`: Description of desired result
- `seed`: Random seed
- `steps`: Inference steps (default: 25)
- `guidance`: Guidance scale (default: 4.0)
- `controlnet_strength`: Strength (default: 0.5)
- `scale_factor`: Upscale multiplier (1.0-4.0, default: 2.0)
- `quantize`: none, 4, or 8

**Outputs:**
- `image`: Upscaled image

---

### Depth & Redux Nodes

#### MFLUX Depth Conditioning
Generates images conditioned on depth information.

**Inputs:**
- `depth_image`: Depth map (or will be extracted from image)
- `prompt`: Text description
- `seed`: Random seed
- `steps`: Inference steps (default: 25)
- `guidance`: Guidance scale (default: 4.0)
- `quantize`: none, 4, or 8

**Outputs:**
- `image`: Generated image matching depth structure

**Tips:**
- Use with MfluxDepthExtraction to get depth from images
- Great for maintaining scene geometry while changing appearance

#### MFLUX Redux (Style Reference)
Generates images using style reference from 1-3 images.

**Inputs:**
- `reference_image1`: First style reference (required)
- `redux_strength1`: Strength for first reference (0.0-2.0, default: 1.0)
- `reference_image2` (optional): Second style reference
- `redux_strength2` (optional): Strength for second reference
- `reference_image3` (optional): Third style reference
- `redux_strength3` (optional): Strength for third reference
- `prompt`: Text description
- `seed`: Random seed
- `steps`: Inference steps (default: 25)
- `guidance`: Guidance scale (default: 4.0)
- `quantize`: none, 4, or 8

**Outputs:**
- `image`: Generated image matching reference style(s)

**Tips:**
- Use multiple references to blend styles
- Adjust strengths to balance influence of each reference
- Strength > 1.0 amplifies style influence

#### MFLUX Depth Extraction
Extracts depth maps from images using DepthPro.

**Inputs:**
- `image`: Input image
- `model_path` (optional): Custom model path

**Outputs:**
- `depth_map`: Depth map image (grayscale)

**Tips:**
- Use output with MfluxDepthConditioning node
- DepthPro is fast and accurate (Apple's model)

---

### Upscaling Nodes

#### MFLUX SeedVR2 Upscale
Upscales images using SeedVR2 model (best quality upscaling).

**Inputs:**
- `model`: SeedVR2 model from loader
- `image`: Image to upscale
- `prompt`: Description of desired result
- `seed`: Random seed
- `steps`: Inference steps (default: 25)
- `guidance`: Guidance scale (default: 4.0)
- `upscale_factor`: 2x, 3x, or 4x (default: 2x)
- `target_resolution` (optional): Specific pixel resolution (overrides factor)

**Outputs:**
- `image`: Upscaled image

**Tips:**
- SeedVR2 provides best quality upscaling
- Use descriptive prompts for better detail enhancement
- Target resolution must be multiple of 16

---

### Fill/Inpainting Nodes

#### MFLUX Fill
Fills or inpaints masked regions in images.

**Inputs:**
- `image`: Input image
- `mask`: Mask image (white=fill, black=preserve)
- `prompt`: Description of what to fill
- `seed`: Random seed
- `steps`: Inference steps (default: 25)
- `guidance`: Guidance scale (default: 4.0)
- `quantize`: none, 4, or 8
- `negative_prompt` (optional): Things to avoid

**Outputs:**
- `image`: Image with filled regions

**Tips:**
- Use white mask for areas to regenerate
- Black mask preserves original content
- Works well for object removal and insertion

---

### LoRA Nodes

#### MFLUX LoRA Loader
Loads and applies up to 4 LoRA models to a base model.

**Inputs:**
- `model`: Base MFLUX model
- `lora_path_1-4` (optional): Paths to LoRA files or HuggingFace repos
- `lora_scale_1-4`: Scale factors (-2.0 to 2.0, default: 1.0)

**Outputs:**
- `model`: Model with LoRAs applied

**Tips:**
- Supports local paths, HuggingFace repos, and collections
- Use negative scales to invert LoRA effects
- Scales > 1.0 amplify LoRA influence

---

### Utility Nodes

#### MFLUX Save Quantized Model
Saves a quantized model to disk for faster loading in future sessions.

**Inputs:**
- `model`: Loaded model to save
- `save_path`: Directory path to save model

**Outputs:**
- None (shows success message)

**Tips:**
- First load takes time, subsequent loads are instant
- Saves disk space with quantization
- Great for frequently used model/quantization combinations

#### MFLUX Config
Configures global MFLUX settings.

**Inputs:**
- `low_ram_mode`: Enable memory optimization (default: false)
- `battery_saver`: Stop generation on low battery (macOS only, default: false)
- `battery_stop_threshold`: Battery percentage to stop at (0-100, default: 20)
- `cache_dir` (optional): Override model cache directory

**Outputs:**
- `config`: Configuration object (for future use)

**Tips:**
- Low RAM mode reduces memory usage at slight speed cost
- Battery saver only works on macOS

---

## Example Workflows

### Basic Text-to-Image
```
MFLUX FLUX.2 Klein Loader (4B, quantize 8)
    ↓
MFLUX Text to Image (prompt, steps=4, guidance=1.0)
    ↓
Save Image
```

### Image Editing
```
Load Image
    ↓
MFLUX FLUX.2 Klein Loader (4B, quantize 8)
    ↓
MFLUX FLUX.2 Edit (edit prompt, steps=4)
    ↓
Save Image
```

### Style Transfer with Redux
```
Load Image (style reference)
    ↓
MFLUX FLUX.1 Loader (dev-redux, quantize 8)
    ↓
MFLUX Redux (prompt, reference image)
    ↓
Save Image
```

### Upscaling
```
Load Image
    ↓
MFLUX SeedVR2 Loader (quantize 8)
    ↓
MFLUX SeedVR2 Upscale (2x, prompt describing details)
    ↓
Save Image
```

### ControlNet Canny
```
Load Image (line art or photo)
    ↓
MFLUX ControlNet Canny (prompt, controlnet_strength=0.8)
    ↓
Save Image
```

### Inpainting
```
Load Image + Load Mask
    ↓
MFLUX Fill (prompt describing fill content)
    ↓
Save Image
```

### Depth-Guided Generation
```
Load Image
    ↓
MFLUX Depth Extraction
    ↓
MFLUX Depth Conditioning (new prompt)
    ↓
Save Image
```

---

## Tips and Best Practices

### Model Selection
- **Fast generation**: FLUX.2 Klein 4B, Z-Image Turbo
- **Best quality**: FLUX.1 dev, Qwen Image
- **Image editing**: FLUX.2 Klein, Qwen Image Edit
- **Upscaling**: SeedVR2
- **Multilingual**: Qwen Image

### Quantization
- **8-bit**: Best balance of speed and quality (recommended)
- **4-bit**: Maximum speed, lowest memory, slight quality loss
- **none**: Best quality but slowest and highest memory

### Guidance Scale
- **FLUX.2 Klein**: Use 1.0 (optimized for this value)
- **Z-Image Turbo**: Use 0.0 (turbo model)
- **Other models**: 3.5-4.0 works well
- **Higher values** (5-10): More prompt adherence, less creativity
- **Lower values** (1-3): More creative, less literal

### Steps
- **FLUX.2 Klein**: 4-8 steps
- **Z-Image Turbo**: 4-9 steps
- **Other models**: 20-30 steps
- **More steps**: Better quality but slower

### Memory Management
- Use 8-bit quantization for most tasks
- Enable low_ram_mode in config for memory-constrained systems
- Save quantized models after first load for faster subsequent loads
- Close other applications when generating large images

### Prompt Writing
- Be specific and descriptive
- Use negative prompts to avoid unwanted elements
- For FIBO: Can use JSON-structured prompts
- For Qwen: Supports multiple languages

---

## Troubleshooting

### Out of Memory
- Use 8-bit or 4-bit quantization
- Reduce image dimensions
- Enable low_ram_mode
- Use smaller models (FLUX.2 4B, Z-Image Turbo)

### Models Not Loading
- Check MFLUX is installed: `pip install mflux`
- Verify internet connection (first download)
- Check cache directory permissions
- Set MFLUX_CACHE_DIR environment variable if needed

### Slow Generation
- Use quantization (8-bit recommended)
- Use faster models (FLUX.2, Z-Image Turbo)
- Reduce number of steps
- Save quantized models for reuse

### Poor Quality Results
- Increase number of steps
- Adjust guidance scale
- Try different models
- Refine prompts to be more specific
- Use higher quantization or none

---

## Requirements

- Python 3.10+
- ComfyUI
- MFLUX (`pip install mflux`)
- MLX framework (automatically installed with MFLUX)
- PyTorch (for ComfyUI image handling)
- PIL/Pillow

---

## License

This integration follows the MFLUX license. Please see the main MFLUX repository for details.

---

## Contributing

Contributions are welcome! Please submit issues and pull requests to the main MFLUX repository.

---

## Credits

- **MFLUX**: Filip Strand
- **ComfyUI Integration**: Community contribution
- **Models**: Black Forest Labs, Alibaba (Qwen), Tongyi-MAI (Z-Image), BRIA.ai (FIBO), Apple (DepthPro)

---

## Support

For issues with:
- **MFLUX functionality**: See main MFLUX repository
- **ComfyUI integration**: Open an issue in the MFLUX repository with "ComfyUI" label
- **ComfyUI itself**: See ComfyUI documentation

---

## Changelog

### Version 1.0.0 (2026-01-20)
- Initial release
- Full support for all MFLUX models and features
- 25+ custom nodes covering all use cases
- Complete documentation and examples
