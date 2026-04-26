"""
segmentation.py
---------------
Two-stage person segmentation:
  1. YOLO detects the bounding box(es) of every person in the frame.
  2. SAM2 takes those boxes as prompts and produces pixel-accurate masks.

The function returns two masks, which the rest of the pipeline uses for
different inpainting targets:
  - bg_mask:     True where the PERSON is → used to preserve the person
                 while the background is painted over.
  - person_mask: 255 where the BACKGROUND is → used as the inpainting mask
                 when we want the model to fill in the background.
"""

import numpy as np
from PIL import Image
from ultralytics import YOLO
from sam2.sam2_image_predictor import SAM2ImagePredictor


def get_person_masks(
    image_path: str,
    image_pil: Image.Image,
    predictor: SAM2ImagePredictor,
    yolo_model_path: str = "yolo26n.pt",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect all people in the image and build a combined segmentation mask.

    Args:
        image_path:      Path to the image on disk (for YOLO).
        image_pil:       The same image as a PIL object (for SAM2).
        predictor:       A pre-loaded SAM2ImagePredictor instance.
        yolo_model_path: Path or name of the YOLO weights to use.

    Returns:
        bg_mask     – boolean array, shape (H, W). True = person pixel.
        person_mask – uint8 array,  shape (H, W). 255 = background pixel.
                      (Inverted so it can be used directly as an inpainting mask.)
    """

    # --- Step 1: YOLO person detection ---
    # class 0 in COCO is "person"; we restrict detection to that class only
    model_yolo = YOLO(yolo_model_path)
    results = model_yolo(image_path, classes=[0])

    # Tell SAM2 which image we are about to segment
    predictor.set_image(image_pil)

    # Start with an empty mask the same size as the image
    h, w = image_pil.size[1], image_pil.size[0]
    master_mask = np.zeros((h, w), dtype=bool)

    # --- Step 2: SAM2 mask prediction per detected bounding box ---
    for result in results:
        if len(result.boxes) == 0:
            print("  No person detected in this frame – skipping segmentation.")
            continue

        for box in result.boxes:
            # xyxy gives [x_min, y_min, x_max, y_max] in pixel coords
            input_box = box.xyxy[0].cpu().numpy()

            masks, scores, _ = predictor.predict(
                box=input_box,
                multimask_output=False,   # single best mask per box
            )

            # Accumulate masks from multiple people with a logical OR
            master_mask = np.logical_or(master_mask, masks[0])

        print("  SAM2 mask generated successfully.")

    # bg_mask:     person=True,  background=False
    bg_mask = master_mask

    # person_mask: person=0,    background=255
    # (white = area the inpainting model is allowed to paint over)
    person_mask = (~master_mask).astype(np.uint8) * 255

    return bg_mask, person_mask
