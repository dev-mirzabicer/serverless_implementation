# Diffusion-Pipe Serverless Endpoint

This is a serverless version of the [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) LoRA training framework, adapted for RunPod Serverless deployment.

## Overview

This converts the interactive pod-based training workflow into a stateless API endpoint. Instead of SSHing into a pod and running an interactive script, you make an HTTP request with your training configuration and receive results when training completes.

## Supported Models

- **Flux** - Requires HuggingFace token
- **SDXL** - Stable Diffusion XL
- **Wan 1.3B** - Wan2.1 Text-to-Video 1.3B
- **Wan 14B T2V** - Wan2.1 Text-to-Video 14B
- **Wan 14B I2V** - Wan2.1 Image-to-Video 14B
- **Qwen Image** - Qwen Image model
- **Z Image Turbo** - Z Image Turbo model

## Building the Docker Image

```bash
# Navigate to the serverless implementation directory
cd serverless_implementation

# Build the image (replace YOUR_USERNAME with your Docker Hub username)
docker build --platform linux/amd64 -t YOUR_USERNAME/diffusion-pipe-serverless:v1.0 .

# Push to Docker Hub
docker push YOUR_USERNAME/diffusion-pipe-serverless:v1.0
```

## API Usage

### Request Format

POST to your endpoint URL with JSON body:

```json
{
  "input": {
    "model_type": "wan14b_t2v",
    "dataset": {
      "type": "videos",
      "videos_url": "https://your-storage.com/my_videos.zip",
      "trigger_word": "ohwx person"
    },
    "training": {
      "epochs": 100,
      "learning_rate": 2e-5,
      "lora_rank": 32,
      "save_every_n_epochs": 10
    },
    "api_keys": {
      "gemini_api_key": "YOUR_GEMINI_API_KEY"
    },
    "output": {
      "upload_url": "https://presigned-s3-url.com/upload"
    }
  }
}
```

### Input Parameters

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | string | One of: `flux`, `sdxl`, `wan13`, `wan14b_t2v`, `wan14b_i2v`, `qwen`, `z_image_turbo` |
| `dataset.type` | string | One of: `images`, `videos`, `both`, `precaptioned` |

#### Dataset Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset.images_url` | string | - | URL to download image dataset (zip file) |
| `dataset.videos_url` | string | - | URL to download video dataset (zip file) |
| `dataset.trigger_word` | string | - | Trigger word to prepend to captions |
| `dataset.image_repeats` | int | 1 | Dataset repeats per epoch for images |
| `dataset.video_repeats` | int | 5 | Dataset repeats per epoch for videos |

#### Training Configuration

The training section supports **all parameters** from diffusion-pipe's TOML config. Any parameter you pass will be included in the generated config. Here are the common ones:

**Basic Training**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `training.epochs` | int | 100 | Number of training epochs |
| `training.learning_rate` | float | 2e-5 | Learning rate |
| `training.batch_size` | int | 1 | Micro batch size per GPU |
| `training.gradient_accumulation_steps` | int | 4 | Gradient accumulation steps |
| `training.gradient_clipping` | float | 1.0 | Gradient norm clipping |
| `training.warmup_steps` | int | 100 | Learning rate warmup steps |

**LoRA Settings**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `training.lora_rank` | int | 32 | LoRA adapter rank (4, 8, 16, 32, 64, 128) |
| `training.lora_dtype` | string | bfloat16 | LoRA weight data type |

**Saving & Checkpoints**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `training.save_every_n_epochs` | int | 10 | Save model every N epochs |
| `training.checkpoint_every_n_minutes` | int | 120 | Save training state for resume |
| `training.save_dtype` | string | bfloat16 | Data type for saved weights |

**Optimizer (AdamW)**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `training.optimizer_type` | string | adamw_optimi | Optimizer type |
| `training.optimizer_betas` | array | [0.9, 0.99] | Adam beta parameters |
| `training.optimizer_weight_decay` | float | 0.01 | Weight decay |
| `training.optimizer_eps` | float | 1e-8 | Adam epsilon |

**Video Training**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `training.video_clip_mode` | string | single_middle | Video clip extraction mode |
| `training.frame_buckets` | array | [1, 33] | Frame buckets for video training |

**Other**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `training.resolution` | int | 1024 | Training resolution (512, 768, 1024) |
| `training.activation_checkpointing` | bool | true | Save VRAM with activation checkpointing |
| `training.caching_batch_size` | int | 1 | Batch size for latent caching |

**Custom Parameters**: Any additional parameters you include in `training` will be passed through to the TOML config.

#### API Keys

| Field | Description |
|-------|-------------|
| `api_keys.huggingface_token` | Required for Flux model |
| `api_keys.gemini_api_key` | Required for video captioning (when type is `videos` or `both`) |

#### Output Configuration (Download Link Generation)

The endpoint automatically generates a download link for your trained LoRA. Multiple strategies are supported:

| Method | Format | Max Size | Retention | Setup Required |
|--------|--------|----------|-----------|----------------|
| `huggingface` (recommended) | Raw .safetensors | Unlimited | Permanent | HF token |
| `s3` | Zip archive | Unlimited | 7 days | Your S3 credentials |
| `litterbox` | Zip archive | 1GB | 72 hours | None |
| `transfer_sh` | Zip archive | 10GB | 14 days | None |
| `auto` (default) | Varies | Varies | Varies | Tries HF → litterbox → transfer.sh |

**Recommended: Upload to HuggingFace Hub (Raw Files)**
```json
{
  "output": {
    "method": "huggingface",
    "huggingface": {
      "token": "hf_YOUR_WRITE_TOKEN",
      "repo_id": "your-username/my-lora",
      "private": true
    }
  }
}
```

This uploads raw `.safetensors` files directly to HuggingFace - no zipping! Perfect for sharing or using your LoRA.

**Simple Usage (No Setup Required):**
```json
{
  "output": {
    "method": "auto"
  }
}
```

**With Your Own S3 (AWS S3, Cloudflare R2, etc.):**
```json
{
  "output": {
    "method": "s3",
    "s3": {
      "endpoint_url": "https://your-account.r2.cloudflarestorage.com",
      "bucket": "my-lora-outputs",
      "region": "auto",
      "access_key": "YOUR_ACCESS_KEY",
      "secret_key": "YOUR_SECRET_KEY",
      "key_prefix": "diffusion-pipe-outputs"
    }
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `output.method` | No | Upload method: `huggingface`, `s3`, `litterbox`, `transfer_sh`, or `auto` |
| `output.huggingface.token` | For HF | HuggingFace token with write access |
| `output.huggingface.repo_id` | No | Repository ID (auto-generated if not provided) |
| `output.huggingface.private` | No | Whether repo should be private (default: true) |
| `output.s3.endpoint_url` | For R2/MinIO | S3 endpoint (optional for AWS S3) |
| `output.s3.bucket` | For S3 | Bucket name |
| `output.s3.region` | For S3 | AWS region or 'auto' for R2 |
| `output.s3.access_key` | For S3 | Access key ID |
| `output.s3.secret_key` | For S3 | Secret access key |

### Response Format

**When using HuggingFace (recommended):**
```json
{
  "status": "success",
  "job_id": "abc123",
  "model_type": "qwen",
  "model_name": "Qwen Image",
  "epochs_completed": 2,
  "latest_epoch": "epoch2",
  "output": {
    "download_url": "https://huggingface.co/username/my-lora/resolve/main/adapter_model.safetensors?download=true",
    "safetensors_url": "https://huggingface.co/username/my-lora/resolve/main/adapter_model.safetensors?download=true",
    "repo_url": "https://huggingface.co/username/my-lora",
    "download_method": "huggingface",
    "expires_in_days": null,
    "note": "Direct download link for your trained LoRA safetensors file"
  }
}
```

**When using other methods (S3, litterbox, etc.):**
```json
{
  "status": "success",
  "job_id": "abc123",
  "model_type": "wan14b_t2v",
  "model_name": "Wan 14B T2V",
  "epochs_completed": 10,
  "latest_epoch": "epoch100",
  "output": {
    "download_url": "https://litter.catbox.moe/abc123.zip",
    "download_method": "litterbox",
    "expires_in_days": 3,
    "archive_size_mb": 256.5,
    "network_volume_path": "/runpod-volume/diffusion_pipe_working_folder/training_outputs/abc123/epoch100"
  }
}
```

The `download_url` is a direct link you can use to download your trained LoRA. Just click it or use `curl`/`wget`.

**Progress Tracking via HuggingFace**: When using HuggingFace output, progress updates are written to the `UPDATES/` folder in your repo. This persists even after RunPod purges job results, so you can always check the status and get the download link from your HuggingFace repository.

### Example Usage with curl

```bash
# Using the async /run endpoint (recommended for long training jobs)
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "model_type": "wan14b_t2v",
      "dataset": {
        "type": "precaptioned",
        "videos_url": "https://my-bucket.s3.amazonaws.com/training_videos.zip"
      },
      "training": {
        "epochs": 50,
        "lora_rank": 32
      }
    }
  }'

# Response:
# {"id": "job-abc123", "status": "IN_QUEUE"}

# Check status
curl https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/job-abc123 \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

## Dataset Preparation

### For Images

1. Create a zip file containing your images
2. Each image should have a corresponding `.txt` caption file (same name, `.txt` extension)
3. If using `type: "images"` without precaptioned data, the system will auto-caption using JoyCaption

### For Videos

1. Create a zip file containing your videos (MP4, AVI, MOV, MKV, WebM)
2. Each video should have a corresponding `.txt` caption file
3. If using `type: "videos"` without precaptioned data, the system will auto-caption using Gemini API

### Precaptioned Data

If your dataset already has captions, use `type: "precaptioned"` to skip the captioning step.

## Network Volume

For faster model loading and to persist trained LoRAs:

1. Create a Network Volume in RunPod
2. Attach it to your serverless endpoint
3. Pre-download models to `/runpod-volume/diffusion_pipe_working_folder/models/`

This significantly reduces cold start time for large models.

## Deployment on RunPod

1. Go to RunPod Serverless console
2. Create a new endpoint
3. Select "Import from Docker Registry"
4. Enter your image: `YOUR_USERNAME/diffusion-pipe-serverless:v1.0`
5. Configure GPU type (H100 recommended for large models)
6. Optionally attach a Network Volume
7. Deploy

## Local Testing

```bash
# Build locally
docker build -t diffusion-pipe-serverless .

# Run with GPU
docker run --gpus all -it diffusion-pipe-serverless

# Test with the test_input.json
# (Inside the container)
python3 /serverless/handler.py
```

## Troubleshooting

### CUDA Errors
- Ensure you're using CUDA 12.8 compatible GPUs
- H100/H200 GPUs are recommended

### Model Download Issues
- Check your HuggingFace token (for Flux)
- Ensure network volume has sufficient space

### Captioning Errors
- Verify Gemini API key is valid (for video captioning)
- Check that images/videos are in supported formats

## Credits

- Original [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) by tdrussell
- RunPod pod template by [Hearmeman24](https://github.com/Hearmeman24/runpod-diffusion_pipe)
- Serverless adaptation for API-based training
