# Diffusion-Pipe Serverless: LoRA Training as an API

Train LoRA adapters for image and video diffusion models via a simple HTTP API. No SSH, no interactive scripts - just send a request and get your trained model.

## What This Does

This project converts [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) (an interactive LoRA training framework) into a **RunPod Serverless endpoint**. Instead of:

```
SSH into pod → Run script → Answer questions → Wait → Download files
```

You now do:

```
Send HTTP request → Get job ID → Poll for status → Download from URL
```

## Features

- **7 Supported Models**: Flux, SDXL, Wan 1.3B, Wan 14B T2V, Wan 14B I2V, Qwen Image, Z Image Turbo
- **Auto-Captioning**: JoyCaption for images, Gemini for videos
- **Direct HuggingFace Upload**: Raw `.safetensors` files, not zipped archives
- **Progress Tracking**: JSON updates written to HuggingFace repo (survives RunPod job purging)
- **Full Parameter Control**: Every diffusion-pipe TOML parameter is configurable via API
- **Multiple Output Options**: HuggingFace (recommended), S3, litterbox, transfer.sh

---

## Quick Start

### 1. Deploy the Endpoint

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Select **Import from Docker Registry**
4. Enter: `mirzabicer/diffusion-pipe-serverless:v1.0`
5. Select GPU type (A100 or H100 recommended)
6. Optionally attach a Network Volume for faster model loading
7. Deploy and note your **Endpoint ID**

### 2. Get Your API Keys

- **RunPod API Key**: [RunPod Settings](https://www.runpod.io/console/user/settings)
- **HuggingFace Token** (for output): [HuggingFace Tokens](https://huggingface.co/settings/tokens) - needs **write** access
- **Gemini API Key** (for video captioning): [Google AI Studio](https://aistudio.google.com/app/apikey)

### 3. Prepare Your Dataset

Create a zip file with your training data:

```
my_dataset.zip
├── image1.jpg
├── image1.txt      # Caption: "a photo of ohwx person smiling"
├── image2.png
├── image2.txt      # Caption: "ohwx person standing outdoors"
└── ...
```

Upload to any publicly accessible URL (litterbox, S3, etc.).

### 4. Start Training

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "model_type": "qwen",
      "dataset": {
        "type": "precaptioned",
        "images_url": "https://your-storage.com/my_dataset.zip"
      },
      "training": {
        "epochs": 100,
        "lora_rank": 32,
        "save_every_n_epochs": 20
      },
      "output": {
        "method": "huggingface",
        "huggingface": {
          "token": "hf_YOUR_WRITE_TOKEN",
          "repo_id": "your-username/my-lora",
          "private": true
        }
      }
    }
  }'
```

Response:
```json
{"id": "abc123-def456", "status": "IN_QUEUE"}
```

### 5. Monitor Progress

**Option A: Poll RunPod Status**
```bash
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/abc123-def456" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

**Option B: Check HuggingFace UPDATES folder**

Your repo will have an `UPDATES/` folder with JSON files:
```
UPDATES/
├── 0001_20241214_120000_INITIALIZED.json
├── 0002_20241214_120030_DOWNLOADING.json
├── 0003_20241214_120145_TRAINING.json
└── 0004_20241214_130000_COMPLETE.json
```

### 6. Download Your LoRA

When complete, the response includes:
```json
{
  "output": {
    "download_url": "https://huggingface.co/your-username/my-lora/resolve/main/adapter_model.safetensors?download=true",
    "repo_url": "https://huggingface.co/your-username/my-lora"
  }
}
```

Just click the URL or use `wget`/`curl` to download.

---

## Supported Models

| Model | `model_type` | Best For | Requirements |
|-------|-------------|----------|--------------|
| **Qwen Image** | `qwen` | Image generation | - |
| **Flux** | `flux` | High-quality images | HuggingFace token |
| **SDXL** | `sdxl` | Stable Diffusion XL | - |
| **Wan 1.3B** | `wan13` | Fast video generation | - |
| **Wan 14B T2V** | `wan14b_t2v` | High-quality text-to-video | - |
| **Wan 14B I2V** | `wan14b_i2v` | Image-to-video | - |
| **Z Image Turbo** | `z_image_turbo` | Fast image generation | - |

---

## Complete API Reference

### Request Structure

```json
{
  "input": {
    "model_type": "string (required)",
    "dataset": { ... },
    "training": { ... },
    "api_keys": { ... },
    "output": { ... }
  }
}
```

### Dataset Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | string | Yes | - | `images`, `videos`, `both`, or `precaptioned` |
| `images_url` | string | For images | - | URL to zip file containing images |
| `videos_url` | string | For videos | - | URL to zip file containing videos |
| `trigger_word` | string | No | - | Word to prepend to all captions (e.g., "ohwx person") |
| `image_repeats` | int | No | 1 | Times to repeat image dataset per epoch |
| `video_repeats` | int | No | 5 | Times to repeat video dataset per epoch |

**Dataset Types Explained:**
- `precaptioned`: Your zip contains `.txt` files alongside media files (recommended)
- `images`: Auto-caption images using JoyCaption
- `videos`: Auto-caption videos using Gemini API (requires `gemini_api_key`)
- `both`: Auto-caption both images and videos

### Training Configuration

Every diffusion-pipe parameter is supported. Here are the most common ones:

#### Basic Training
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epochs` | int | 100 | Total training epochs |
| `learning_rate` | float | 2e-5 | Optimizer learning rate |
| `batch_size` | int | 1 | Micro batch size per GPU |
| `gradient_accumulation_steps` | int | 4 | Steps before weight update |
| `gradient_clipping` | float | 1.0 | Max gradient norm |
| `warmup_steps` | int | 100 | LR warmup steps |

#### LoRA Settings
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lora_rank` | int | 32 | Rank of LoRA matrices (4, 8, 16, 32, 64, 128) |
| `lora_dtype` | string | "bfloat16" | LoRA weight precision |

#### Checkpointing
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save_every_n_epochs` | int | 10 | Save model every N epochs |
| `checkpoint_every_n_minutes` | int | 120 | Save training state for resume |
| `save_dtype` | string | "bfloat16" | Saved model precision |

#### Optimizer (AdamW)
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `optimizer_type` | string | "adamw_optimi" | Optimizer class |
| `optimizer_betas` | array | [0.9, 0.99] | Adam beta parameters |
| `optimizer_weight_decay` | float | 0.01 | L2 regularization |
| `optimizer_eps` | float | 1e-8 | Numerical stability |

#### Video-Specific
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video_clip_mode` | string | "single_middle" | How to extract clips |
| `frame_buckets` | array | [1, 33] | Frame count buckets |

#### Other
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `resolution` | int | 1024 | Training resolution |
| `activation_checkpointing` | bool | true | Save VRAM |
| `caching_batch_size` | int | 1 | Latent caching batch size |
| `eval_every_n_epochs` | int | 1 | Evaluation frequency |
| `eval_before_first_step` | bool | true | Eval at start |

**Custom Parameters**: Any field not listed above will be passed directly to the TOML config. This means you can use ANY parameter that diffusion-pipe supports.

### API Keys

| Field | Required For | Description |
|-------|-------------|-------------|
| `huggingface_token` | Flux model | HuggingFace access token |
| `gemini_api_key` | Video captioning | Google Gemini API key |

### Output Configuration

#### HuggingFace (Recommended)
```json
{
  "output": {
    "method": "huggingface",
    "huggingface": {
      "token": "hf_YOUR_WRITE_TOKEN",
      "repo_id": "username/repo-name",
      "private": true
    }
  }
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `token` | Yes | - | HuggingFace token with write access |
| `repo_id` | No | Auto-generated | Repository name (e.g., "user/my-lora") |
| `private` | No | true | Make repository private |

**Benefits:**
- Raw `.safetensors` files (not zipped)
- Permanent storage
- Progress tracking via UPDATES folder
- Direct download URLs

#### S3-Compatible Storage
```json
{
  "output": {
    "method": "s3",
    "s3": {
      "endpoint_url": "https://account.r2.cloudflarestorage.com",
      "bucket": "my-bucket",
      "region": "auto",
      "access_key": "YOUR_ACCESS_KEY",
      "secret_key": "YOUR_SECRET_KEY",
      "key_prefix": "lora-outputs"
    }
  }
}
```

#### Auto (No Setup Required)
```json
{
  "output": {
    "method": "auto"
  }
}
```
Tries: HuggingFace (if configured) → litterbox → transfer.sh → file.io

---

## Progress Tracking (HuggingFace)

When using HuggingFace output, progress is written to `UPDATES/` as JSON files:

### File Naming
```
UPDATES/{number}_{timestamp}_{status}.json
```
Example: `0003_20241214_143052_TRAINING.json`

### JSON Structure
```json
{
  "status": "TRAINING",
  "timestamp": "2024-12-14T14:30:52.123456+00:00",
  "job_id": "abc123-def456",
  "update_number": 3,
  "repo_id": "username/my-lora",
  "data": {
    "message": "Starting Qwen Image LoRA training",
    "epochs": 100,
    "lora_rank": 32
  }
}
```

### Status Values

| Status | Meaning |
|--------|---------|
| `INITIALIZED` | Repository created, job starting |
| `STARTING` | Validating input, preparing |
| `DOWNLOADING` | Downloading dataset |
| `CAPTIONING` | Running auto-captioning |
| `MODEL_DOWNLOAD` | Downloading base model |
| `TRAINING` | Training in progress |
| `TRAINING_COMPLETE` | Training finished |
| `UPLOADING` | Uploading results |
| `COMPLETE` | All done - includes download URLs |
| `ERROR` | Something failed |

### Final COMPLETE Update
```json
{
  "status": "COMPLETE",
  "timestamp": "2024-12-14T15:45:00.000000+00:00",
  "job_id": "abc123-def456",
  "update_number": 8,
  "repo_id": "username/my-lora",
  "data": {
    "message": "Training completed successfully!",
    "repository_url": "https://huggingface.co/username/my-lora",
    "direct_download_url": "https://huggingface.co/username/my-lora/resolve/main/adapter_model.safetensors?download=true",
    "instructions": "Use the direct_download_url to download your trained LoRA safetensors file.",
    "files_uploaded": ["adapter_model.safetensors", "adapter_config.json"],
    "safetensors_files": ["adapter_model.safetensors"]
  }
}
```

### Building an App with Progress Tracking

```python
from huggingface_hub import HfApi
import json
import time

def monitor_training(repo_id: str, token: str):
    api = HfApi(token=token)
    seen_updates = set()

    while True:
        # List files in UPDATES folder
        files = api.list_repo_files(repo_id, repo_type="model")
        update_files = sorted([f for f in files if f.startswith("UPDATES/")])

        # Process new updates
        for update_file in update_files:
            if update_file not in seen_updates:
                seen_updates.add(update_file)

                # Download and parse
                content = api.hf_hub_download(repo_id, update_file, repo_type="model")
                with open(content) as f:
                    update = json.load(f)

                print(f"[{update['status']}] {update['data'].get('message', '')}")

                # Check if complete
                if update['status'] == 'COMPLETE':
                    return update['data']['direct_download_url']
                elif update['status'] == 'ERROR':
                    raise Exception(update['data'].get('error', 'Training failed'))

        time.sleep(30)  # Poll every 30 seconds

# Usage
download_url = monitor_training("username/my-lora", "hf_token")
print(f"Download your LoRA: {download_url}")
```

---

## Response Format

### Success Response
```json
{
  "status": "success",
  "job_id": "abc123-def456",
  "model_type": "qwen",
  "model_name": "Qwen Image",
  "epochs_completed": 100,
  "latest_epoch": "epoch100",
  "output": {
    "download_url": "https://huggingface.co/.../adapter_model.safetensors?download=true",
    "safetensors_url": "https://huggingface.co/.../adapter_model.safetensors?download=true",
    "repo_url": "https://huggingface.co/username/my-lora",
    "download_method": "huggingface",
    "expires_in_days": null,
    "note": "Direct download link for your trained LoRA safetensors file"
  }
}
```

### Error Response
```json
{
  "status": "failed",
  "error": "Description of what went wrong"
}
```

---

## Examples

### Train Qwen Image LoRA (Simplest)
```bash
curl -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer RUNPOD_KEY" \
  -d '{
    "input": {
      "model_type": "qwen",
      "dataset": {
        "type": "precaptioned",
        "images_url": "https://example.com/images.zip"
      },
      "output": {
        "method": "huggingface",
        "huggingface": {
          "token": "hf_xxx",
          "repo_id": "user/qwen-lora"
        }
      }
    }
  }'
```

### Train Wan 14B Video LoRA with Custom Settings
```bash
curl -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer RUNPOD_KEY" \
  -d '{
    "input": {
      "model_type": "wan14b_t2v",
      "dataset": {
        "type": "precaptioned",
        "videos_url": "https://example.com/videos.zip",
        "video_repeats": 10
      },
      "training": {
        "epochs": 200,
        "learning_rate": 1e-5,
        "lora_rank": 64,
        "save_every_n_epochs": 25,
        "gradient_accumulation_steps": 8,
        "video_clip_mode": "multiple_overlapping",
        "frame_buckets": [1, 17, 33, 49]
      },
      "output": {
        "method": "huggingface",
        "huggingface": {
          "token": "hf_xxx",
          "repo_id": "user/wan-video-lora",
          "private": false
        }
      }
    }
  }'
```

### Train Flux with Auto-Captioning
```bash
curl -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer RUNPOD_KEY" \
  -d '{
    "input": {
      "model_type": "flux",
      "dataset": {
        "type": "images",
        "images_url": "https://example.com/raw_images.zip",
        "trigger_word": "ohwx person"
      },
      "training": {
        "epochs": 80,
        "lora_rank": 32
      },
      "api_keys": {
        "huggingface_token": "hf_xxx_for_flux_download"
      },
      "output": {
        "method": "huggingface",
        "huggingface": {
          "token": "hf_xxx_for_upload",
          "repo_id": "user/flux-lora"
        }
      }
    }
  }'
```

---

## Dataset Preparation

### Image Dataset Structure
```
dataset.zip
├── photo1.jpg
├── photo1.txt          # "a portrait photo of ohwx person"
├── photo2.png
├── photo2.txt          # "ohwx person walking in a park"
├── subfolder/          # Subfolders are flattened automatically
│   ├── photo3.webp
│   └── photo3.txt
└── ...
```

**Supported formats:** JPG, JPEG, PNG, WebP, BMP, TIFF

### Video Dataset Structure
```
dataset.zip
├── clip1.mp4
├── clip1.txt           # "ohwx person dancing in a studio"
├── clip2.mov
├── clip2.txt           # "close-up of ohwx person talking"
└── ...
```

**Supported formats:** MP4, AVI, MOV, MKV, WebM

### Caption File Format

Each caption should be on a single line:
```
a professional photo of ohwx person wearing a blue shirt, studio lighting, white background
```

For trigger words, you can either:
1. Include them in every caption manually
2. Use the `trigger_word` parameter to auto-prepend

### Tips for Good Results

1. **Consistent quality**: Use similar resolution/quality across images
2. **Varied poses/angles**: Include diversity in your dataset
3. **Good captions**: Be specific and consistent with style
4. **Right amount**: 10-50 images is usually enough for faces/characters
5. **No macOS artifacts**: The system auto-removes `__MACOSX` and `.DS_Store`

---

## Network Volume (Recommended)

Using a Network Volume dramatically reduces cold start time by caching models.

### Setup
1. Create a Network Volume in RunPod
2. Attach it to your serverless endpoint
3. First run will download models to `/runpod-volume/diffusion_pipe_working_folder/models/`
4. Subsequent runs reuse cached models

### Pre-downloading Models

To avoid download time during training, you can pre-populate models:

```bash
# SSH into a pod with the network volume attached
# Models will be stored at:
# /runpod-volume/diffusion_pipe_working_folder/models/
```

---

## Troubleshooting

### "Model type X is not implemented"
- Check that `model_type` is exactly one of: `flux`, `sdxl`, `wan13`, `wan14b_t2v`, `wan14b_i2v`, `qwen`, `z_image_turbo`

### "No images/videos found in directory"
- Ensure your zip file contains actual media files
- Check that files aren't nested too deeply in folders
- macOS users: Create zip via terminal (`zip -r`) not Finder

### "Failed to download dataset"
- Verify your URL is publicly accessible
- Test with `curl YOUR_URL` locally first

### "HuggingFace upload failed"
- Ensure your token has **write** access
- Check token at https://huggingface.co/settings/tokens

### Training is slow
- Use H100 or A100 GPUs for best performance
- Attach a Network Volume to cache models
- Reduce `epochs` for testing

### Out of memory
- Reduce `batch_size` to 1
- Enable `activation_checkpointing` (default: true)
- Use lower `lora_rank` (16 or 32)

---

## Pricing Estimate

RunPod charges per-second for GPU time. Rough estimates:

| Model | GPU | ~Time for 100 epochs | ~Cost |
|-------|-----|---------------------|-------|
| Qwen Image | A100 80GB | 10-20 min | $0.50-1.00 |
| Flux | A100 80GB | 15-30 min | $0.75-1.50 |
| Wan 14B | H100 | 30-60 min | $2.00-4.00 |

Actual costs depend on dataset size, resolution, and configuration.

---

## Credits

- [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) by tdrussell - Original training framework
- [runpod-diffusion_pipe](https://github.com/Hearmeman24/runpod-diffusion_pipe) by Hearmeman24 - RunPod pod template
- Serverless adaptation for API-based training

---

## License

This project adapts open-source tools for serverless deployment. Please respect the licenses of the underlying projects.
