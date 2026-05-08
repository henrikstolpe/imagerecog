#!/bin/bash
# Full IR Recognition Pipeline: Generate → Train → Inference
# Runs inside Docker container with GPU access
set -e

echo "=============================================="
echo "IR SIGNATURE RECOGNITION - FULL PIPELINE"
echo "=============================================="
echo ""

# Check GPU
echo "=== GPU Check ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Step 1: Generate dataset (smaller for faster demo)
echo "=== Step 1: Generating Synthetic IR Dataset ==="
python3 scripts/generate_dataset.py \
    --output /app/data/signature_db \
    --num-angles 8 \
    --images-per-angle 4 \
    --seed 42
echo ""

# Step 2: Train model
echo "=== Step 2: Training Model (QLoRA on PaliGemma 2 3B) ==="
python3 scripts/train.py \
    --database /app/data/signature_db \
    --checkpoint /app/checkpoints/best \
    --epochs 10 \
    --batch-size 4 \
    --learning-rate 2e-4
echo ""

# Step 3: Run inference on a generated test image
echo "=== Step 3: Running Inference ==="
# Generate a single test image for inference
python3 -c "
from ir_recognition.ir_generator import IRGenerator
from ir_recognition.models import IRGeneratorConfig
from PIL import Image
import numpy as np

gen = IRGenerator(seed=99)
config = IRGeneratorConfig(vehicle_type='T-72', azimuth=135.0, elevation=30.0)
img = gen.generate(config)
img_uint8 = (img * 255).astype(np.uint8)
Image.fromarray(img_uint8, mode='L').save('/app/data/test_image.png')
print('Generated test image: /app/data/test_image.png')
"

python3 scripts/recognize.py \
    /app/data/test_image.png \
    --checkpoint /app/checkpoints/best \
    --database /app/data/signature_db
echo ""

echo "=============================================="
echo "PIPELINE COMPLETE"
echo "=============================================="
