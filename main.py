"""
main.py
-------
Entry point – orchestrates the full pipeline for every image in IMAGE_DIR.

Run with:
    python main.py

For each input image the pipeline will:
  1. Resize it to 1024×1024.
  2. Extract a face-identity embedding (ArcFace via InsightFace).
  3. Segment the person with YOLO + SAM2.
  4. Load the SDXL + IP-Adapter pipeline.
  5. Generate face → clothing → background in three inpainting passes.
  6. Save the output image and append quality metrics to a CSV file.

All configurable parameters (paths, prompts, seeds, …) are in config.py.
"""

import os
import time
import csv

import torch
import numpy as np
from pathlib import Path
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import (
    IMAGE_DIR, OUTPUT_DIR, CSV_PATH, TEMP_IMAGE,
    SAM2_CHECKPOINT, SAM2_CONFIG,
    BASE_MODEL, IP_CKPT, DEVICE,
    TARGET_SIZE, NUM_OF_IMGS,
    PROMPTS_BACKGROUND, PROMPTS_CLOTHING, PROMPT_PERSON,
)
from face_utils import extract_faceid_embeds, get_face_bbox
from segmentation import get_person_masks
from pipeline_loader import load_pipeline
from generation import generate_scene
from evaluate import func_evaluate



# One-time model loading  
base_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading SAM2 model...")
sam2_model = build_sam2(
    SAM2_CONFIG,
    os.path.join(base_dir, SAM2_CHECKPOINT),
    device=DEVICE,
)
predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 ready.\n")


# Main loop

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect input images (PNG and JPG), process at most NUM_OF_IMGS
    image_paths = sorted(
        list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpg"))
    )[:NUM_OF_IMGS]

    if not image_paths:
        print(f"No images found in {IMAGE_DIR}. Exiting.")
        return

    # Open the CSV in write mode so we start fresh on each run
    with open(CSV_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["filename", "face_identity_score", "clip_score", "brisque_score", "time"],
        )
        writer.writeheader()

        for idx, image_path in enumerate(image_paths):
            print(f"\n{'='*60}")
            print(f"Processing [{idx + 1}/{len(image_paths)}]: {image_path.name}")
            print(f"{'='*60}")

            run_start = time.perf_counter()

            # 1. Preprocess: Resize to TARGET_SIZE × TARGET_SIZE
            print("\n[1/5] Preprocessing image...")

            img = Image.open(image_path).convert("RGB")
            img.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

            # Paste onto a black canvas so the aspect ratio is preserved
            canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
            paste_x = (TARGET_SIZE - img.width)  // 2
            paste_y = (TARGET_SIZE - img.height) // 2
            canvas.paste(img, (paste_x, paste_y))
            canvas.save(TEMP_IMAGE)

            input_path = os.path.join(base_dir, TEMP_IMAGE)
            image_pil  = Image.open(input_path).convert("RGB")

            # Pick matching prompts (cycle through the lists round-robin)
            prompt_bg       = PROMPTS_BACKGROUND[idx % len(PROMPTS_BACKGROUND)]
            prompt_clothing = PROMPTS_CLOTHING[idx % len(PROMPTS_CLOTHING)]

            # 2. Extract face-identity embedding
            print("\n[2/5] Extracting FaceID embedding...")
            faceid_embeds = extract_faceid_embeds(input_path, DEVICE)
            print(f"  Embedding shape: {faceid_embeds.shape}")


            # 3. Segment the person (YOLO → SAM2)
            print("\n[3/5] Segmenting person...")
            bg_mask, person_mask = get_person_masks(
                image_path=input_path,
                image_pil=image_pil,
                predictor=predictor,
            )

            # 4. Load generation pipeline
            print("\n[4/5] Loading generation pipeline...")
            ip_model = load_pipeline(BASE_MODEL, IP_CKPT, DEVICE)

            # We need the face bounding box before calling generate_scene
            face_bbox = get_face_bbox(input_path)
            if face_bbox is None:
                print("  WARNING: No face found – skipping this image.")
                continue


            # 5. Generate: face → clothing → background
            print("\n[5/5] Running 3-step generation...")
            output_image = generate_scene(
                ip_model=ip_model,
                image=image_pil,
                faceid_embeds=faceid_embeds,
                face_bbox=face_bbox,
                bg_mask=bg_mask,
                person_mask=person_mask,
                prompt_person=PROMPT_PERSON,
                prompt_clothing=prompt_clothing,
                prompt_background=prompt_bg,
            )

            # Save output and evaluate
            output_path = OUTPUT_DIR / f"{image_path.stem}_output.png"
            output_image.save(output_path)

            elapsed = time.perf_counter() - run_start
            print(f"\n  Saved → {output_path}  ({elapsed:.1f}s)")

            full_prompt = f"{prompt_bg}, {PROMPT_PERSON}, {prompt_clothing}"
            face_score, clip_score, brisque_score = func_evaluate(
                image_pil, output_image, full_prompt
            )

            writer.writerow({
                "filename":           image_path.stem,
                "face_identity_score": face_score,
                "clip_score":          clip_score,
                "brisque_score":       brisque_score.item(),
                "time":                elapsed,
            })
            csv_file.flush()   # write immediately in case of a crash mid-run

            print(
                f"  Metrics — "
                f"FaceID: {face_score:.4f}  |  "
                f"CLIP: {clip_score:.4f}  |  "
                f"BRISQUE: {brisque_score:.4f}"
            )

    print(f"\nAll done. Results saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
