# Diffusion-Pipe Serverless: Developer Guide

This document provides comprehensive information for developers who want to understand, modify, extend, or contribute to this codebase.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [How It Works](#how-it-works)
5. [Adding New Models](#adding-new-models)
6. [Modifying TOML Generation](#modifying-toml-generation)
7. [Extending Output Handlers](#extending-output-handlers)
8. [Progress Tracking System](#progress-tracking-system)
9. [Local Development](#local-development)
10. [Testing](#testing)
11. [Debugging](#debugging)
12. [Code Patterns](#code-patterns)
13. [Common Issues](#common-issues)

---

## Architecture Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RunPod Serverless                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HTTP Request                                                                │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐     │
│  │  handler.py │───▶│  Download Data   │───▶│  Run Captioning        │     │
│  │             │    │  (zip from URL)  │    │  (JoyCaption/Gemini)   │     │
│  └─────────────┘    └──────────────────┘    └─────────────────────────┘     │
│       │                                              │                       │
│       ▼                                              ▼                       │
│  ┌─────────────────────┐    ┌────────────────────────────────────────┐      │
│  │  Generate TOMLs     │───▶│  Run Training (deepspeed train.py)     │      │
│  │  (model + dataset)  │    │  diffusion-pipe framework              │      │
│  └─────────────────────┘    └────────────────────────────────────────┘      │
│                                              │                               │
│                                              ▼                               │
│                             ┌────────────────────────────────────────┐      │
│                             │  output_handler.py                     │      │
│                             │  - Upload to HuggingFace (raw files)   │      │
│                             │  - Or S3/litterbox (zip archive)       │      │
│                             │  - Return download URL                 │      │
│                             └────────────────────────────────────────┘      │
│                                              │                               │
│                                              ▼                               │
│                                      JSON Response                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Stateless Design**: Each job is independent. All state (models, datasets) is either downloaded fresh or loaded from a network volume.

2. **Network Volume for Persistence**: Models are cached on `/runpod-volume` to avoid re-downloading large base models (20-40GB) on every job.

3. **HuggingFace Progress Tracking**: Since RunPod purges job results after some time, we write progress to HuggingFace repos for persistent tracking.

4. **Dynamic TOML Generation**: Instead of static config files, we generate TOMLs from JSON input, allowing any diffusion-pipe parameter to be set via the API.

5. **Priority-Based Output**: Multiple upload strategies with fallbacks (HuggingFace → litterbox → transfer.sh → file.io).

---

## Project Structure

```
serverless_implementation/
├── Dockerfile              # Docker build configuration
├── handler.py              # Main serverless handler (entry point)
├── output_handler.py       # Upload and progress tracking logic
├── start_serverless.sh     # Container startup script
├── input_schema.json       # JSON Schema for API validation
├── test_input.json         # Example test payload
├── README.md               # Original README (brief)
├── README_NEW.md           # Comprehensive user documentation
└── DEVELOPERS.md           # This file

# From base image (hearmeman/diffusion-pipe:v11):
/diffusion_pipe/            # diffusion-pipe training framework
/serverless/                # Our serverless code (copied during build)
/serverless/Captioning/     # Captioning scripts (JoyCaption, video_captioner)
```

---

## Core Components

### handler.py

The main entry point. Contains:

| Function | Purpose |
|----------|---------|
| `handler(job)` | Main RunPod handler - orchestrates entire workflow |
| `validate_input(input_data)` | Validates JSON input against schema |
| `download_and_extract_dataset(url, dest, job)` | Downloads and extracts zip datasets |
| `flatten_directory(dir_path)` | Handles nested zips and removes macOS artifacts |
| `run_image_captioning(trigger_word, job)` | Runs JoyCaption for images |
| `run_video_captioning(gemini_key, job)` | Runs Gemini API for videos |
| `download_model(model_type, hf_token, job)` | Downloads or verifies base model |
| `generate_model_toml(input_data, model_type, job_id)` | Generates training config |
| `generate_dataset_toml(input_data, model_type)` | Generates dataset config |
| `run_training(model_type, job, job_id)` | Executes deepspeed training |

**Key Constants:**
```python
NETWORK_VOLUME = "/runpod-volume"
WORKING_DIR = f"{NETWORK_VOLUME}/diffusion_pipe_working_folder"
DIFFUSION_PIPE_DIR = f"{WORKING_DIR}/diffusion_pipe"
OUTPUT_DIR = f"{WORKING_DIR}/training_outputs"
IMAGE_DATASET_DIR = f"{WORKING_DIR}/image_dataset_here"
VIDEO_DATASET_DIR = f"{WORKING_DIR}/video_dataset_here"
```

**Model Configuration:**
```python
MODEL_CONFIGS = {
    "flux": {
        "model_name": "Flux",
        "model_path": "models/flux",
        "requires_hf_token": True,
    },
    "qwen": {
        "model_name": "Qwen Image",
        "model_path": "models/Qwen-Image",
        "toml_type": "qwen_image",  # IMPORTANT: Maps to diffusion-pipe's type
    },
    # ... etc
}
```

### output_handler.py

Handles uploading results and progress tracking. Contains:

| Class/Function | Purpose |
|----------------|---------|
| `HuggingFaceProgressTracker` | Writes JSON progress updates to HF repo |
| `OutputHandler` | Main class for upload logic |
| `OutputHandler.upload_to_huggingface()` | Uploads raw files to HuggingFace Hub |
| `OutputHandler.upload_to_litterbox()` | Uploads zip to litterbox.catbox.moe |
| `OutputHandler.upload_to_transfer_sh()` | Uploads zip to transfer.sh |
| `OutputHandler.upload_to_user_s3()` | Uploads zip to user's S3 bucket |
| `OutputHandler.process_output()` | Main method - tries upload strategies in priority order |
| `upload_and_get_download_url()` | Convenience wrapper |
| `init_progress_tracker()` | Initializes global progress tracker |
| `get_progress_tracker()` | Gets global progress tracker |

### start_serverless.sh

Startup script that:
1. Detects network volume path
2. Creates directory structure
3. Copies diffusion_pipe and Captioning scripts to working directory
4. Installs flash-attention from prebuilt wheel
5. Starts the Python handler

---

## How It Works

### 1. Request Processing

```python
def handler(job: Dict) -> Dict:
    job_id = job["id"]
    input_data = job.get("input", {})

    # 1. Validate input
    error = validate_input(input_data)
    if error:
        return {"status": "failed", "error": error}

    # 2. Initialize progress tracker (if HuggingFace configured)
    hf_config = output_config.get("huggingface")
    if hf_config and hf_config.get("token"):
        progress_tracker = init_progress_tracker(hf_config, job_id)

    # 3. Download datasets
    # 4. Run captioning (if needed)
    # 5. Download model (if not cached)
    # 6. Generate TOML configs
    # 7. Run training
    # 8. Upload results
    # 9. Return response
```

### 2. Dataset Handling

The system handles several dataset types:

| Type | Description | Processing |
|------|-------------|------------|
| `images` | Images without captions | Download → JoyCaption → Train |
| `videos` | Videos without captions | Download → Gemini caption → Train |
| `both` | Images and videos | Download both → Caption both → Train |
| `precaptioned` | Data with existing .txt captions | Download → Train (skip captioning) |

**Zip Extraction Logic:**
```python
def flatten_directory(dir_path):
    """
    Handles:
    1. macOS __MACOSX artifacts (removed)
    2. .DS_Store files (removed)
    3. Single nested folder (flattened)

    Example: data.zip containing data/image1.jpg → extracts to dest/image1.jpg
    """
```

### 3. TOML Generation

The `generate_model_toml()` function creates diffusion-pipe config dynamically:

```python
def generate_model_toml(input_data, model_type, job_id):
    training = input_data.get("training", {})

    # Defaults (verified from actual diffusion-pipe TOML files)
    defaults = {
        "epochs": 100,
        "learning_rate": 2e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        # ... etc
    }

    # User values override defaults
    config = {**defaults, **training}

    # Generate TOML content
    content = f"""
output_dir = '{output_path}'
dataset = 'examples/dataset.toml'
epochs = {config['epochs']}
# ... etc

[model]
type = '{model_type_for_toml}'
# Model-specific config

[adapter]
type = 'lora'
rank = {config['lora_rank']}

[optimizer]
type = 'adamw_optimi'
lr = {config['learning_rate']}
"""

    # Custom parameters: anything not in standard_params gets added
    custom_params = {k: v for k, v in training.items() if k not in standard_params}
    for key, value in custom_params.items():
        content += f"{key} = {to_toml_value(value)}\n"
```

### 4. Training Execution

```python
def run_training(model_type, job, job_id):
    cmd = [
        "deepspeed",
        "--num_gpus=1",
        "train.py",
        "--deepspeed",
        "--config", f"examples/job_{job_id}.toml"
    ]

    process = subprocess.Popen(
        cmd,
        cwd=DIFFUSION_PIPE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # ...
    )

    # Stream output, look for epoch progress, detect completion
```

### 5. Output Directory Structure

diffusion-pipe creates:
```
training_outputs/{job_id}/{timestamp}/
├── epoch1/                    # ← safetensors (for inference)
│   ├── adapter_model.safetensors
│   └── adapter_config.json
├── epoch2/
│   └── ...
└── global_step{N}/            # ← DeepSpeed checkpoints (for resume only)
    ├── layer_00-model_states.pt
    ├── layer_01-model_states.pt
    └── ... (63 files)
```

**Important**: We upload `epoch{N}/` directories (contain safetensors), NOT `global_step{N}/` (DeepSpeed checkpoints).

---

## Adding New Models

To add support for a new model:

### Step 1: Add to MODEL_CONFIGS

```python
# In handler.py
MODEL_CONFIGS = {
    # ... existing models ...
    "new_model": {
        "model_name": "Human-Readable Name",
        "model_path": "models/NewModel",  # Path relative to WORKING_DIR
        "requires_hf_token": False,        # True if needs HF authentication
        "toml_type": "new_model_type",     # Type string for diffusion-pipe
    },
}
```

### Step 2: Add Model Download Logic

```python
# In download_model() function
elif model_type == "new_model":
    # Check if model exists
    model_path = os.path.join(WORKING_DIR, MODEL_CONFIGS[model_type]["model_path"])
    if os.path.exists(model_path):
        log(f"Model already exists at {model_path}")
        return True

    # Download logic
    # Option A: From HuggingFace
    subprocess.run([
        "huggingface-cli", "download",
        "organization/model-name",
        "--local-dir", model_path
    ])

    # Option B: From direct URL
    # Option C: Already in base Docker image
```

### Step 3: Add TOML Generation

```python
# In generate_model_toml() function
elif model_type == "new_model":
    content += f"""type = 'new_model_type'
diffusers_path = '{model_path}'
dtype = 'bfloat16'
# Add model-specific config here
"""
```

### Step 4: Update input_schema.json

```json
{
  "model_type": {
    "type": "string",
    "enum": ["flux", "sdxl", "wan13", "wan14b_t2v", "wan14b_i2v", "qwen", "z_image_turbo", "new_model"]
  }
}
```

### Step 5: Test

1. Build Docker image locally
2. Test with a small dataset
3. Verify TOML generation is correct
4. Verify output directory structure

---

## Modifying TOML Generation

### Understanding the Structure

diffusion-pipe TOML files have these sections:

```toml
# Root level - training settings
output_dir = '/path/to/output'
dataset = 'path/to/dataset.toml'
epochs = 100
micro_batch_size_per_gpu = 1
gradient_accumulation_steps = 4
# ... more training params

[model]
type = 'model_type'
diffusers_path = '/path/to/model'
dtype = 'bfloat16'
# ... model-specific params

[adapter]
type = 'lora'
rank = 32
dtype = 'bfloat16'

[optimizer]
type = 'adamw_optimi'
lr = 2e-5
betas = [0.9, 0.99]
weight_decay = 0.01
eps = 1e-8
```

### Adding New Parameters

To add a new parameter:

1. **Add to defaults** (if it has a sensible default):
```python
defaults = {
    # ... existing ...
    "new_param": default_value,
}
```

2. **Add to content generation**:
```python
content += f"new_param = {to_toml_value(config.get('new_param', default_value))}\n"
```

3. **Add to standard_params** (to prevent it being added as custom):
```python
standard_params = {
    # ... existing ...
    'new_param',
}
```

4. **Update input_schema.json**:
```json
"new_param": {
    "type": "number",
    "default": 1.0,
    "description": "Description here"
}
```

### The to_toml_value Helper

```python
def to_toml_value(value):
    """Convert Python values to TOML format."""
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, list):
        items = ", ".join(str(to_toml_value(v)) if isinstance(v, (bool, str)) else str(v) for v in value)
        return f"[{items}]"
    else:
        return str(value)
```

---

## Extending Output Handlers

### Adding a New Upload Method

1. **Add the upload method** in `OutputHandler` class:

```python
def upload_to_new_service(self, file_path: str) -> Optional[str]:
    """
    Upload file to NewService and return download URL.

    Args:
        file_path: Path to the file to upload

    Returns:
        Download URL or None if failed
    """
    try:
        # Your upload logic here
        response = requests.post(
            'https://api.newservice.com/upload',
            files={'file': open(file_path, 'rb')},
            timeout=3600
        )

        if response.status_code == 200:
            return response.json()['download_url']
        return None

    except Exception as e:
        logger.error(f"NewService upload error: {e}")
        return None
```

2. **Add to process_output()** priority chain:

```python
# In process_output() method
if not download_url and method in ('new_service', 'auto'):
    logger.info("Attempting NewService upload...")
    download_url = self.upload_to_new_service(archive_path)
    if download_url:
        result["download_method"] = "new_service"
        result["expires_in_days"] = 30  # or whatever
```

3. **Update input_schema.json**:

```json
"method": {
    "enum": ["auto", "huggingface", "s3", "litterbox", "transfer_sh", "new_service"]
}
```

---

## Progress Tracking System

### HuggingFaceProgressTracker Class

```python
class HuggingFaceProgressTracker:
    def __init__(self, hf_config: Dict, job_id: str):
        self.hf_config = hf_config
        self.job_id = job_id
        self.repo_id = None
        self.api = None
        self.initialized = False
        self.update_count = 0

    def initialize(self) -> bool:
        """Create repo and UPDATES folder."""

    def update(self, status: str, message: str, **kwargs) -> bool:
        """Write a progress update."""

    def complete(self, download_url: str, safetensors_url: str, **kwargs) -> bool:
        """Write final completion update."""

    def error(self, error_message: str, **kwargs) -> bool:
        """Write error update."""
```

### JSON Update Format

```json
{
    "status": "TRAINING",
    "timestamp": "2024-12-14T23:59:59.123456+00:00",
    "job_id": "abc123-def456",
    "update_number": 3,
    "repo_id": "username/my-lora",
    "data": {
        "message": "Training started",
        "epochs": 100,
        "lora_rank": 32
    }
}
```

### Status Values

| Status | When Used |
|--------|-----------|
| `INITIALIZED` | Repo created, before any processing |
| `STARTING` | Job initialization |
| `DOWNLOADING` | Downloading dataset |
| `CAPTIONING` | Running captioning |
| `MODEL_DOWNLOAD` | Downloading base model |
| `TRAINING` | Training in progress |
| `TRAINING_COMPLETE` | Training finished |
| `UPLOADING` | Uploading to HuggingFace |
| `COMPLETE` | All done, includes download URLs |
| `ERROR` | Something failed |

---

## Local Development

### Prerequisites

- Docker with BuildKit
- Access to a machine with NVIDIA GPU (for testing)
- HuggingFace account (for testing uploads)

### Building Locally

```bash
cd serverless_implementation

# Build image
docker build --platform linux/amd64 -t diffusion-pipe-serverless:dev .

# Build takes ~5 minutes (mostly copying base image layers)
```

### Testing Locally (with GPU)

```bash
# Run container with GPU
docker run --gpus all -it \
    -v /path/to/test/data:/data \
    diffusion-pipe-serverless:dev bash

# Inside container:
cd /serverless
python3 handler.py  # Uses test_input.json
```

### Testing Without GPU

For testing everything except actual training:

```bash
# Mock the training function
# Edit handler.py to skip run_training() and return mock result
```

### Development Workflow

1. Make changes to Python files locally
2. Copy to remote machine: `scp *.py remote:/tmp/serverless_implementation/`
3. SSH and rebuild: `docker build ... && docker push ...`
4. Kill RunPod worker to pull new image
5. Test via API

---

## Testing

### Unit Testing (Not Yet Implemented)

```python
# Suggested structure for future tests
def test_flatten_directory():
    """Test that macOS artifacts are removed and nested folders flattened."""

def test_generate_model_toml():
    """Test TOML generation for each model type."""

def test_validate_input():
    """Test input validation."""
```

### Integration Testing

Use small datasets and few epochs:

```json
{
    "input": {
        "model_type": "qwen",
        "dataset": {
            "type": "precaptioned",
            "images_url": "https://your-small-test-dataset.zip"
        },
        "training": {
            "epochs": 2,
            "save_every_n_epochs": 1
        }
    }
}
```

### Verifying TOML Generation

```python
# Add this temporarily to handler.py
toml_content = generate_model_toml(input_data, model_type, job_id)
print("Generated TOML:")
print(toml_content)
# Compare against working manual TOML
```

---

## Debugging

### Common Log Locations

In RunPod logs:
- Startup: Look for "Diffusion-Pipe Serverless Worker Starting..."
- Job start: `Starting job {job_id}`
- Progress: `[INFO] Progress: ...`
- Training output: Streamed from deepspeed
- Errors: `[ERROR] ...`

### Debugging Dataset Issues

```python
# After extraction, log directory contents
def download_and_extract_dataset(...):
    # ... extraction code ...

    # Debug: list what was extracted
    for root, dirs, files in os.walk(dest):
        log(f"Directory: {root}")
        log(f"  Files: {files[:10]}...")  # First 10 files
```

### Debugging TOML Issues

If training fails immediately, check the generated TOML:

```bash
# In container
cat /runpod-volume/diffusion_pipe_working_folder/diffusion_pipe/examples/job_*.toml
```

### Debugging Upload Issues

```python
# Add verbose logging
def upload_to_huggingface(...):
    logger.info(f"Output dir contents: {os.listdir(output_dir)}")
    for root, dirs, files in os.walk(output_dir):
        logger.info(f"  {root}: {files}")
```

---

## Code Patterns

### Error Handling Pattern

```python
# Consistent pattern used throughout
try:
    result = do_something()
    if not result:
        if progress_tracker:
            progress_tracker.error("Something failed")
        return {"status": "failed", "error": "Something failed"}
except Exception as e:
    log(f"Error: {e}", "ERROR")
    if progress_tracker:
        progress_tracker.error(str(e))
    return {"status": "failed", "error": str(e)}
```

### Progress Update Pattern

```python
# Before starting an operation
if progress_tracker:
    progress_tracker.update("STATUS", "Human message", key=value)

# Do the operation
result = do_operation()

# Handle failure
if not result:
    if progress_tracker:
        progress_tracker.error("Operation failed")
    return failure_response
```

### Configuration Merge Pattern

```python
# Defaults + user overrides
defaults = {"key": "default_value"}
config = {**defaults, **user_config}
value = config.get("key", fallback)
```

---

## Common Issues

### Issue: "Model type X is not implemented"

**Cause**: The `toml_type` in MODEL_CONFIGS doesn't match what diffusion-pipe expects.

**Solution**: Check diffusion-pipe's train.py for valid model types. For Qwen, it's `qwen_image`, not `qwen`.

### Issue: "Directory had no images/videos"

**Cause**: Zip extraction failed or macOS artifacts weren't cleaned.

**Solution**: Check `flatten_directory()` is working. Look for `__MACOSX` in logs.

### Issue: Training completes but no safetensors

**Cause**: Handler is selecting `global_step` directories instead of `epoch` directories.

**Solution**: The handler should prioritize `epoch` directories. Check `run_training()` return logic.

### Issue: HuggingFace upload fails silently

**Cause**: Token doesn't have write permission or repo creation failed.

**Solution**:
1. Verify token at https://huggingface.co/settings/tokens
2. Token needs "Write" permission
3. Check if repo_id format is valid (username/repo-name)

### Issue: Progress updates not appearing

**Cause**: Progress tracker not initialized or failed silently.

**Solution**: Check logs for "Initializing HuggingFace progress tracker" and any errors.

---

## Contributing

### Code Style

- Use type hints where practical
- Log important operations with `log()` function
- Handle errors gracefully, always returning structured responses
- Update progress tracker at each significant step

### Pull Request Checklist

- [ ] Tested locally with Docker
- [ ] Tested on RunPod with real job
- [ ] Updated input_schema.json if adding parameters
- [ ] Updated README_NEW.md if adding features
- [ ] Updated this DEVELOPERS.md if changing architecture

### Version Bumping

When making changes:
1. Update Dockerfile label version
2. Tag Docker image with new version
3. Document changes in commit message

---

## References

- [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) - Original training framework
- [RunPod Serverless Docs](https://docs.runpod.io/serverless) - RunPod serverless documentation
- [HuggingFace Hub API](https://huggingface.co/docs/huggingface_hub) - HuggingFace upload API
- [DeepSpeed](https://www.deepspeed.ai/) - Training optimization framework
