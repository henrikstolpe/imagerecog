# Image Recognition System

A fine-tunable image recognition system built on PaliGemma 2 (3B) with QLoRA. Train it on your own image categories and deploy a web interface for classification and similarity search. Military vehicles (T-72, Challenger 2, Leopard 2) are used as an example here, but the system works with any image categories.

## How It Works

### Architecture

The system uses Google's PaliGemma 2 3B vision-language model as a backbone — specifically the **SigLIP-So400m vision encoder** (a Vision Transformer) to extract visual features from images. On top of this encoder, two lightweight heads are trained:

1. **Classification Head** — A linear layer that maps the encoder's 1152-dimensional output to class probabilities. Trained with cross-entropy loss.

2. **Embedding Head** — A linear layer that maps to a 256-dimensional L2-normalized vector. Used for similarity search against a database of known images. Trained with triplet margin loss.

Both heads share the same vision encoder, so a single forward pass produces both a classification and an embedding.

### QLoRA Fine-Tuning

The base PaliGemma 2 model is 3 billion parameters — far too large to fully fine-tune on a consumer GPU. Instead, we use **QLoRA**:

- The base model is loaded in **4-bit quantization** (NF4 format via bitsandbytes), reducing memory from ~12GB to ~1.5GB
- **LoRA adapters** (rank 16, alpha 32) are applied to the attention projection layers (q, k, v, out) of the vision encoder
- Only the LoRA parameters (~4M) and the two heads are trained — the rest of the 3B model stays frozen
- Peak VRAM usage during training: ~1.5GB (fits easily on an 8GB GPU)

### Training Process

1. **Data Import** — Images are organized in folders by class (one subfolder per category). They're resized to 224×224 and stored in a signature database with a manifest.

2. **Split Generation** — Images are split into train (70%), validation (15%), and test (15%) sets, stratified by class.

3. **Batch Construction** — The triplet loss requires batches with at least 2 images per class. A custom `TripletBatchSampler` ensures each batch contains images from at least 2 different classes with 2 samples each.

4. **Training Loop** — Each epoch:
   - Forward pass through the quantized vision encoder + LoRA adapters
   - Classification head produces logits → cross-entropy loss
   - Embedding head produces vectors → triplet margin loss (with online hard mining)
   - Combined loss = 0.5 × classification + 0.5 × triplet
   - Backpropagation updates only LoRA weights and head parameters

5. **Checkpoint Saving** — After training, the LoRA weights, classification head, embedding head, and config (including class names and their order) are saved.

6. **Embedding Computation** — All database images are passed through the trained model to pre-compute their embeddings for fast similarity search at inference time.

### Recognition (Inference)

When you upload an image:

1. **Preprocessing** — The image is loaded, converted to RGB, resized to 224×224, and normalized.

2. **Forward Pass** — The image goes through the quantized vision encoder (with trained LoRA weights) producing a feature vector.

3. **Classification** — The classification head maps features to class probabilities via softmax. The top-3 predictions with confidence scores are returned.

4. **Similarity Search** — The embedding head produces a 256-d vector. This is compared (cosine similarity) against all pre-computed database embeddings. The top-5 most similar images are returned with their scores.

5. **Threshold Flags**:
   - `low_confidence` — Set when the top-1 classification confidence is below 50%
   - `no_match` — Set when all similarity scores are below 30%

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX 4060, 8GB VRAM)
- Docker with NVIDIA Container Toolkit
- Hugging Face account with access to [google/paligemma2-3b-pt-224](https://huggingface.co/google/paligemma2-3b-pt-224)

### 1. Prepare Training Images

The `data.zip` archive contains the expected file structure with example images you can use as a reference. Extract it to get started:

```bash
unzip data.zip
```

Organize your images in folders — one subfolder per category:

```
data/real_images/
├── category-a/
│   ├── img001.jpeg
│   └── ...
├── category-b/
│   └── ...
└── category-c/
    └── ...
```

Aim for 50+ images per class with variety in angles, lighting, and backgrounds.

### 2. Build the Docker Image

```bash
docker build -t image-recognition .
```

### 3. Train the Model

```bash
docker run --gpus all --rm \
  -e HF_TOKEN=your_huggingface_token \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/data/real_images:/app/data/real_images \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  image-recognition python3 scripts/train_real_images.py \
    --images /app/data/real_images \
    --epochs 15 \
    --learning-rate 3e-4
```

The first run downloads PaliGemma 2 (~6GB). Subsequent runs use the cached model. Training takes ~2-3 minutes for 150 images with 15 epochs.

### 4. Run the Web UI

```bash
docker run --gpus all --rm -p 8000:8000 \
  -e HF_TOKEN=your_huggingface_token \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/data/real_signature_db:/app/data/real_signature_db \
  -v $(pwd)/checkpoints/real:/app/checkpoints/real \
  image-recognition python3 -m ir_recognition.web \
    --model-path /app/checkpoints/real \
    --db-path /app/data/real_signature_db \
    --port 8000
```

Open http://localhost:8000 and upload an image to classify.

## Project Structure

```
├── src/ir_recognition/
│   ├── training/          # Model loading, QLoRA, trainer, dataloader
│   ├── inference/         # Pipeline, preprocessing, embedding computation
│   ├── signature_db/      # Image database with manifest and splits
│   ├── web/               # FastAPI backend + vanilla JS frontend
│   └── models.py          # Dataclasses (configs, results, metadata)
├── scripts/
│   ├── train_real_images.py   # Train on your own images
│   ├── recognize.py           # CLI inference
│   └── run_pipeline.sh        # Full pipeline (generate→train→infer)
├── data/
│   ├── real_images/           # Your training images (by class folder)
│   └── real_signature_db/     # Generated database + embeddings
├── checkpoints/real/          # Trained model weights
├── Dockerfile
└── pyproject.toml
```

## Tips for Better Accuracy

- **More images** — 100+ per class significantly improves generalization
- **Image variety** — Different angles, distances, lighting, backgrounds
- **More epochs** — Monitor val_loss; stop when it starts increasing (overfitting)
- **Similarity search** — Even when classification is uncertain, similarity search against the database is often correct since it compares directly against known examples
