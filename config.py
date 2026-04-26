"""
config.py
---------
Central place for all the settings and creative prompts used across the pipeline.
"""

from pathlib import Path


# Paths
IMAGE_DIR  = Path("test_images/")          # folder containing input photos
OUTPUT_DIR = Path("output/")               # where generated images and the CSV go
CSV_PATH   = OUTPUT_DIR / "output.csv"
TEMP_IMAGE = "test_input_standardized_2.png"  # scratch file for the resized input

# Model identifiers

BASE_MODEL = "SG161222/RealVisXL_V3.0"        # SDXL-based realistic model
IP_CKPT    = "ip-adapter-faceid_sdxl.bin"     # IP-Adapter weights that inject face identity
DEVICE     = "cuda"

# SAM2 checkpoint
SAM2_CHECKPOINT = "segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_t.yaml"

YOLO_MODEL = "yolo26n.pt"   # YOLO weights used for person detection

# Generation hyper-parameters
SEED            = 42    # fixed seed for reproducibility
NUM_SAMPLES     = 1     # how many images to produce per call
WIDTH           = 1024
HEIGHT          = 1024
TARGET_SIZE     = 1024  # letterbox target before feeding the pipeline
INFERENCE_STEPS = 30    # DDIM steps – more steps = better quality but slower

NUM_OF_IMGS     = 2     # how many input images to process in one run

# Negative prompt (tells the model what to avoid)
NEGATIVE_PROMPT = (
    "monochrome, lowres, bad anatomy, worst quality, "
    "low quality, blurry, deformed"
)

# Person prompt (kept fixed – we just want a clean full-body render)

PROMPT_PERSON = (
    "A realistic full-body photo of a person, "
    "detailed face, natural lighting, sharp focus"
)

# Background prompts  (one per scene; cycled through images round-robin)

PROMPTS_BACKGROUND = [
    "Standing on a tropical beach at sunset, brown sand, soft pink and purple sky, "
    "natural skin tones, balanced lighting, realistic colors",

    "Standing in a forest, depth between subject and background, "
    "soft natural light, realistic colors",

    "Standing in a city at night, buildings with lights, dark sky, "
    "balanced lighting on subject, realistic colors",

    "Standing in a desert, sand dunes visible, blue sky, "
    "strong sunlight, natural lighting, realistic colors",

    "Standing in a modern living room, neutral wall colors, "
    "soft indoor lighting, clean background, realistic interior photography",

    "Standing in an industrial area, large factory buildings in the background, "
    "metal structures and pipes visible, open sky above, natural lighting, realistic colors",

    "Standing on a university campus, academic buildings in the background, "
    "walkways and greenery visible, open sky, natural lighting, realistic colors",

    "Standing on a highway road, road stretching into the distance, "
    "vehicles and horizon line visible, blue sky, natural lighting, realistic colors",

    "Standing on an island shoreline, ocean visible in the background, "
    "rocks and water visible, horizon line distinct, blue sky, natural lighting, realistic colors",

    "Standing in ancient ruins, stone structures in the background, "
    "detailed architecture and textures visible, open sky, natural lighting, realistic colors",

    "Standing on a neon-lit street at night, buildings and glowing signs clearly visible, "
    "dark sky, balanced lighting on subject, realistic colors, no color cast",

    "Standing at a harbor, water and ships visible in the background, "
    "docks and structures visible, open sky, natural lighting, realistic colors",
]


# Clothing prompts  (matched 1-to-1 with background prompts above)

PROMPTS_CLOTHING = [
    "person wearing light summer dress, bright colors, sandals, relaxed beachwear",
    "person wearing casual hiking outfit, earth tones, boots and outdoor jacket",
    "person wearing stylish evening outfit, dark colors, elegant and modern streetwear",
    "person wearing loose linen shirt and trousers, neutral tones, light breathable fabric",
    "person wearing smart casual outfit, neutral colors, clean modern look",
    "person wearing worker outfit, dark sturdy clothing, boots, practical workwear",
    "person wearing smart casual university style, light colors, backpack, clean look",
    "person wearing sporty streetwear, joggers and hoodie, sneakers, casual modern style",
    "person wearing resort wear, light tropical colors, relaxed beach style clothing",
    "person wearing bohemian style clothing, earthy tones, flowy fabric, vintage accessories",
    "person wearing vibrant streetwear, bold colors, urban fashion, modern accessories",
    "person wearing nautical style outfit, navy and white colors, casual maritime clothing",
]
