# Identity-Preserving Scene Generation Pipeline

A deep learning pipeline that takes a portrait photo of a person and generates a fully synthetic scene around them — new background, new clothing — while keeping their face identity intact using **Stable Diffusion XL**, **IP-Adapter FaceID**, **SAM2**, and **YOLO**.

---

## How it works

The pipeline runs three sequential inpainting passes on each input image:

```
Input photo
    │
    ├─ FaceID embedding (InsightFace ArcFace)
    ├─ Person segmentation (YOLO → SAM2)
    │
    ▼
Step 1 – Face refinement      (IP-Adapter scale 0.95, strength 0.40)
Step 2 – Clothing generation  (body region only, face excluded)
    └─ Feathered face blend
Step 3 – Background generation (noise-scramble original BG first)
    │
    ▼
Output image  +  evaluation metrics (CSV)
```

The key idea is that each step targets a **different spatial region** of the image, so the model never has to solve face identity, clothing style, and scene composition all at once.

---

## Project structure

```
project/
│
├── main.py               # Entry point — run this to process images
├── config.py             # All settings, paths, and prompts (edit here)
├── face_utils.py         # ArcFace embedding + face bounding box detection
├── segmentation.py       # YOLO person detection → SAM2 pixel masks
├── pipeline_loader.py    # SDXL inpainting + IP-Adapter model loading
├── generation.py         # 3-step face / clothing / background generation
├── evaluate.py           # Metric functions (FaceID score, CLIP, BRISQUE)
│
├── test_images/          # Put your input images here (.png or .jpg)
├── output/               # Generated images and metrics CSV saved here
│
├── segment-anything-2/
│   └── checkpoints/
│       └── sam2.1_hiera_tiny.pt
└── ip-adapter-faceid_sdxl.bin
```

---

## Requirements

### Python packages

```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install opencv-python pillow numpy
pip install ultralytics          # YOLO
pip install insightface onnxruntime-gpu
pip install ip-adapter           # IP-Adapter FaceID
```

Install SAM2 from source:

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 && pip install -e .
```

### Model checkpoints

| Model | Source |
|---|---|
| SDXL Inpainting | Auto-downloaded via `diffusers` |
| RealVisXL V3.0 | Auto-downloaded via `diffusers` (HuggingFace: `SG161222/RealVisXL_V3.0`) |
| IP-Adapter FaceID SDXL | Download `ip-adapter-faceid_sdxl.bin` from [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) |
| SAM2 (hiera tiny) | Download `sam2.1_hiera_tiny.pt` from the [SAM2 releases](https://github.com/facebookresearch/segment-anything-2/releases) |
| InsightFace buffalo_l | Auto-downloaded on first run |
| YOLO | Place `yolo26n.pt` in the project root |

### Hardware

A CUDA-capable GPU with at least **16 GB VRAM** is recommended. The pipeline loads the SDXL UNet in `float16` to reduce memory usage.

---

## Usage

### 1. Add input images

Drop `.png` or `.jpg` portrait photos into the `test_images/` folder.

### 2. Configure (optional)

Open `config.py` to adjust:

- `NUM_OF_IMGS` — how many images to process per run
- `SEED` — change for different generation results
- `INFERENCE_STEPS` — more steps = better quality, slower runtime
- `PROMPTS_BACKGROUND` / `PROMPTS_CLOTHING` — add or edit scene styles

### 3. Run

```bash
python main.py
```

Generated images are saved to `output/` as `<original_name>_output.png`.
Quality metrics are appended to `output/check_wesi.csv`.

---

## Output metrics

Each processed image gets three scores written to the CSV:

| Metric | What it measures | Good range |
|---|---|---|
| **FaceID score** | Cosine similarity between input and output face embeddings | > 0.5 (same person) |
| **CLIP score** | Alignment between the generated image and the text prompt | > 0.25 |
| **BRISQUE** | No-reference image quality (lower = less distortion) | < 40 |

---

## Design decisions

**Why three separate inpainting passes?**
Trying to change clothing, background, and face simultaneously in one pass causes the model to compromise between objectives. Separating them gives each step a clear, focused target and better results overall.

**Why noise-scramble the background before Step 3?**
If the original background pixels are left intact, the diffusion model tends to follow the original scene composition instead of generating a truly new one. Random noise forces it to start fresh.

**Why low strength (0.40) for the face step?**
High strength would fully replace the face with a hallucinated one. We only want to add sharpness and detail on top of the real face, so a low strength value preserves identity while improving quality.

**Why is IP-Adapter scale set to 0 for clothing and background?**
The face-identity embedding should only influence the face step. Leaving it active during clothing generation causes facial texture to bleed into fabric patterns.

---

## Troubleshooting

**`No face detected`** — InsightFace requires a reasonably clear frontal or near-frontal face. Make sure the subject's face is visible and well-lit in the input image.

**`No person found in image`** — YOLO didn't detect a person. Try a photo where the full body or at least the upper body is visible.

**Out of VRAM** — Reduce `INFERENCE_STEPS` in `config.py`, or try processing one image at a time by setting `NUM_OF_IMGS = 1`.

**Slow runtime** — The pipeline loads the SDXL model fresh for each image. This is intentional for simplicity; for batch processing speed you can move `load_pipeline()` outside the loop in `main.py`.
