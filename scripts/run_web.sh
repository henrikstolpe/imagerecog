#!/bin/bash
# Run the full pipeline then start the web UI
set -e

echo "=============================================="
echo "IR SIGNATURE RECOGNITION - PIPELINE + WEB UI"
echo "=============================================="
echo ""

# Run the full pipeline (generate, train, inference test)
bash scripts/run_pipeline.sh

echo ""
echo "=============================================="
echo "STARTING WEB UI"
echo "=============================================="
echo "  URL: http://localhost:8000"
echo "=============================================="
echo ""

# Start the web UI server
python3 -m ir_recognition.web \
    --model-path /app/checkpoints/best \
    --db-path /app/data/signature_db \
    --host 0.0.0.0 \
    --port 8000
