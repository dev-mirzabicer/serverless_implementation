#!/usr/bin/env python3
"""
Serverless Handler for Diffusion-Pipe LoRA Training

This handler converts the interactive diffusion-pipe training workflow
into a serverless API endpoint for RunPod.
"""

import os
import sys
import json
import subprocess
import shutil
import tempfile
import zipfile
import tarfile
import requests
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import runpod

# Import our output handler module
from output_handler import (
    upload_and_get_download_url,
    init_progress_tracker,
    get_progress_tracker,
    HuggingFaceProgressTracker
)

# ============================================================================
# Configuration Constants
# ============================================================================

# Base paths - these will be set based on environment
NETWORK_VOLUME = os.environ.get("NETWORK_VOLUME", "/runpod-volume")
WORKING_DIR = os.path.join(NETWORK_VOLUME, "diffusion_pipe_working_folder")
DIFFUSION_PIPE_DIR = os.path.join(WORKING_DIR, "diffusion_pipe")

# Dataset directories
IMAGE_DATASET_DIR = os.path.join(WORKING_DIR, "image_dataset_here")
VIDEO_DATASET_DIR = os.path.join(WORKING_DIR, "video_dataset_here")

# Output directory
OUTPUT_DIR = os.path.join(WORKING_DIR, "training_outputs")

# Model configurations
MODEL_CONFIGS = {
    "flux": {
        "toml_file": "flux.toml",
        "model_name": "Flux",
        "requires_hf_token": True,
        "model_path": "models/flux",
        "hf_repo": "black-forest-labs/FLUX.1-dev"
    },
    "sdxl": {
        "toml_file": "sdxl.toml",
        "model_name": "SDXL",
        "requires_hf_token": False,
        "model_path": "models/sdXL_v10VAEFix.safetensors",
        "hf_repo": "timoshishi/sdXL_v10VAEFix",
        "hf_file": "sdXL_v10VAEFix.safetensors"
    },
    "wan13": {
        "toml_file": "wan13_video.toml",
        "model_name": "Wan 1.3B",
        "requires_hf_token": False,
        "model_path": "models/Wan/Wan2.1-T2V-1.3B",
        "hf_repo": "Wan-AI/Wan2.1-T2V-1.3B"
    },
    "wan14b_t2v": {
        "toml_file": "wan14b_t2v.toml",
        "model_name": "Wan 14B T2V",
        "requires_hf_token": False,
        "model_path": "models/Wan/Wan2.1-T2V-14B",
        "hf_repo": "Wan-AI/Wan2.1-T2V-14B"
    },
    "wan14b_i2v": {
        "toml_file": "wan14b_i2v.toml",
        "model_name": "Wan 14B I2V",
        "requires_hf_token": False,
        "model_path": "models/Wan/Wan2.1-I2V-14B-480P",
        "hf_repo": "Wan-AI/Wan2.1-I2V-14B-480P"
    },
    "qwen": {
        "toml_file": "qwen_toml.toml",
        "model_name": "Qwen Image",
        "requires_hf_token": False,
        "model_path": "models/Qwen-Image",
        "hf_repo": "Qwen/Qwen-Image"
    },
    "z_image_turbo": {
        "toml_file": "z_image_toml.toml",
        "model_name": "Z Image Turbo",
        "requires_hf_token": False,
        "model_path": "models/z_image",
        "hf_repo": "Comfy-Org/z_image_turbo"
    }
}

# ============================================================================
# Utility Functions
# ============================================================================

def log(message: str, level: str = "INFO"):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


def send_progress(job: Dict, message: str, progress: Optional[float] = None):
    """Send a progress update to the client."""
    update = {"message": message}
    if progress is not None:
        update["progress"] = progress
    runpod.serverless.progress_update(job, update)
    log(f"Progress: {message}")


def run_command(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict] = None) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    log(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        log(f"Command failed with code {result.returncode}", "ERROR")
        log(f"STDOUT: {result.stdout}", "ERROR")
        log(f"STDERR: {result.stderr}", "ERROR")

    return result


def download_file(url: str, dest_path: str) -> bool:
    """Download a file from a URL."""
    log(f"Downloading from {url} to {dest_path}")
    try:
        response = requests.get(url, stream=True, timeout=3600)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        log(f"Downloaded {downloaded} bytes")
        return True
    except Exception as e:
        log(f"Download failed: {e}", "ERROR")
        return False


def extract_archive(archive_path: str, dest_dir: str) -> bool:
    """Extract a zip or tar archive."""
    log(f"Extracting {archive_path} to {dest_dir}")
    try:
        os.makedirs(dest_dir, exist_ok=True)

        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extractall(dest_dir)
        elif archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as t:
                t.extractall(dest_dir)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as t:
                t.extractall(dest_dir)
        else:
            log(f"Unknown archive format: {archive_path}", "ERROR")
            return False

        log("Extraction complete")
        return True
    except Exception as e:
        log(f"Extraction failed: {e}", "ERROR")
        return False


def flatten_directory(dir_path: str):
    """
    Flatten a directory structure by moving all files to the root.
    This handles the case where archives contain a single top-level folder.
    Also removes macOS artifacts like __MACOSX and .DS_Store.
    """
    # First, remove macOS artifacts
    macosx_dir = os.path.join(dir_path, "__MACOSX")
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir)
        log("Removed __MACOSX directory")

    # Remove .DS_Store files recursively
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f == ".DS_Store":
                os.remove(os.path.join(root, f))

    # Get remaining items (excluding hidden files)
    items = [i for i in os.listdir(dir_path) if not i.startswith('.')]

    # Check if there's only one item and it's a directory
    if len(items) == 1:
        single_item = os.path.join(dir_path, items[0])
        if os.path.isdir(single_item):
            log(f"Flattening single subdirectory: {items[0]}")
            # Move all contents up one level
            for item in os.listdir(single_item):
                if item.startswith('.'):
                    continue  # Skip hidden files
                src = os.path.join(single_item, item)
                dst = os.path.join(dir_path, item)
                shutil.move(src, dst)
            # Remove the now-empty directory
            shutil.rmtree(single_item)

            # Recursively flatten in case of nested structure
            flatten_directory(dir_path)


# ============================================================================
# Dataset Handling
# ============================================================================

def download_and_extract_dataset(url: str, dest_dir: str, job: Dict) -> bool:
    """Download and extract a dataset from a URL."""
    send_progress(job, f"Downloading dataset from {url[:50]}...")

    # Clean the destination directory
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    # Determine file extension from URL
    url_lower = url.lower()
    if '.zip' in url_lower:
        ext = '.zip'
    elif '.tar.gz' in url_lower or '.tgz' in url_lower:
        ext = '.tar.gz'
    elif '.tar' in url_lower:
        ext = '.tar'
    else:
        ext = '.zip'  # Default assumption

    # Download to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        if not download_file(url, temp_path):
            return False

        send_progress(job, "Extracting dataset...")
        if not extract_archive(temp_path, dest_dir):
            return False

        # Flatten if needed
        flatten_directory(dest_dir)

        # Count files
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

        image_count = 0
        video_count = 0

        for f in os.listdir(dest_dir):
            ext_lower = os.path.splitext(f)[1].lower()
            if ext_lower in image_exts:
                image_count += 1
            elif ext_lower in video_exts:
                video_count += 1

        log(f"Dataset extracted: {image_count} images, {video_count} videos")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================================
# Captioning
# ============================================================================

def run_image_captioning(trigger_word: Optional[str], job: Dict) -> bool:
    """Run JoyCaption for image captioning."""
    send_progress(job, "Running image captioning with JoyCaption...")

    captioning_script = os.path.join(WORKING_DIR, "Captioning", "JoyCaption", "JoyCaptionRunner.sh")

    if not os.path.exists(captioning_script):
        log(f"JoyCaption script not found at {captioning_script}", "ERROR")
        return False

    cmd = ["bash", captioning_script]
    if trigger_word:
        cmd.extend(["--trigger-word", trigger_word])
    cmd.append(IMAGE_DATASET_DIR)

    env = {"NETWORK_VOLUME": WORKING_DIR}

    # Run captioning
    result = run_command(cmd, cwd=WORKING_DIR, env=env)

    if result.returncode != 0:
        log("Image captioning failed", "ERROR")
        return False

    # Verify captions were created
    txt_count = sum(1 for f in os.listdir(IMAGE_DATASET_DIR) if f.endswith('.txt'))
    log(f"Created {txt_count} caption files")

    return txt_count > 0


def run_video_captioning(gemini_api_key: str, job: Dict) -> bool:
    """Run Gemini-based video captioning."""
    send_progress(job, "Running video captioning with Gemini...")

    captioning_script = os.path.join(WORKING_DIR, "Captioning", "video_captioner.sh")

    if not os.path.exists(captioning_script):
        log(f"Video captioning script not found at {captioning_script}", "ERROR")
        return False

    env = {
        "NETWORK_VOLUME": WORKING_DIR,
        "GEMINI_API_KEY": gemini_api_key
    }

    result = run_command(["bash", captioning_script], cwd=WORKING_DIR, env=env)

    if result.returncode != 0:
        log("Video captioning failed", "ERROR")
        return False

    # Verify captions were created
    txt_count = sum(1 for f in os.listdir(VIDEO_DATASET_DIR) if f.endswith('.txt'))
    log(f"Created {txt_count} video caption files")

    return txt_count > 0


# ============================================================================
# Configuration Generation
# ============================================================================

def generate_dataset_toml(
    input_data: Dict,
    has_images: bool,
    has_videos: bool
) -> str:
    """Generate the dataset.toml configuration file."""
    training = input_data.get("training", {})
    dataset = input_data.get("dataset", {})

    resolution = training.get("resolution", 1024)
    frame_buckets = training.get("frame_buckets", [1, 33])
    image_repeats = dataset.get("image_repeats", 1)
    video_repeats = dataset.get("video_repeats", 5)

    content = f"""# Auto-generated dataset configuration
resolutions = [{resolution}]

enable_ar_bucket = true
min_ar = 0.5
max_ar = 2.0
num_ar_buckets = 7

frame_buckets = {json.dumps(frame_buckets)}
"""

    if has_images:
        content += f"""
[[directory]]
path = '{IMAGE_DATASET_DIR}'
num_repeats = {image_repeats}
"""

    if has_videos:
        content += f"""
[[directory]]
path = '{VIDEO_DATASET_DIR}'
num_repeats = {video_repeats}
"""

    return content


def generate_model_toml(
    input_data: Dict,
    model_type: str,
    job_id: str
) -> str:
    """
    Generate the model training configuration file.

    The 'training' section supports ANY TOML parameter that diffusion-pipe accepts.
    Common parameters have defaults, but you can override anything or add new ones.

    Example training config:
    {
        "training": {
            "epochs": 100,
            "learning_rate": 2e-5,
            "lora_rank": 32,
            "activation_checkpointing": false,  # Override default
            "video_clip_mode": "random",        # Change video mode
            "custom_param": "value"             # Add any custom param
        }
    }
    """
    model_config = MODEL_CONFIGS[model_type]
    training = input_data.get("training", {})

    # Default values (can be overridden by training config)
    # These match the defaults from diffusion-pipe's example TOML files
    defaults = {
        # Training settings
        "epochs": 100,
        "learning_rate": 2e-5,
        "batch_size": 1,  # maps to micro_batch_size_per_gpu
        "gradient_accumulation_steps": 4,
        "gradient_clipping": 1.0,
        "warmup_steps": 100,
        # Eval settings
        "eval_every_n_epochs": 1,
        "eval_before_first_step": True,
        "eval_micro_batch_size_per_gpu": 1,
        "eval_gradient_accumulation_steps": 1,
        # Misc settings
        "save_every_n_epochs": 10,
        "checkpoint_every_n_minutes": 120,
        "activation_checkpointing": True,
        "partition_method": "parameters",
        "save_dtype": "bfloat16",
        "caching_batch_size": 1,
        "steps_per_print": 1,
        "video_clip_mode": "single_middle",
        # Model dtype settings
        "model_dtype": "bfloat16",          # [model] dtype
        "transformer_dtype": None,           # [model] transformer_dtype (None = not set, use model_dtype)
        # Adapter (LoRA) settings
        "lora_rank": 32,
        "lora_dtype": "bfloat16",            # [adapter] dtype
        # Optimizer settings (adamw_optimi)
        "optimizer_type": "adamw_optimi",
        "optimizer_betas": [0.9, 0.99],
        "optimizer_weight_decay": 0.01,
        "optimizer_eps": 1e-8,
    }

    # Merge user config with defaults (user values override defaults)
    config = {**defaults, **training}

    # Extract known parameters
    epochs = config.get("epochs", 100)
    lr = config.get("learning_rate", 2e-5)
    batch_size = config.get("batch_size", 1)
    grad_accum = config.get("gradient_accumulation_steps", 4)
    lora_rank = config.get("lora_rank", 32)
    save_every = config.get("save_every_n_epochs", 10)

    output_path = os.path.join(OUTPUT_DIR, job_id)
    model_path = os.path.join(WORKING_DIR, model_config["model_path"])

    # Helper function to convert Python values to TOML format
    def to_toml_value(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, list):
            items = ", ".join(str(to_toml_value(v)) if isinstance(v, (bool, str)) else str(v) for v in value)
            return f"[{items}]"
        else:
            return str(value)

    # Build configuration content
    content = f"""# Auto-generated training configuration for {model_config['model_name']}
# Job ID: {job_id}
# Generated: {datetime.now().isoformat()}

output_dir = '{output_path}'
dataset = 'examples/dataset.toml'

# Training settings
epochs = {config.get('epochs', epochs)}
micro_batch_size_per_gpu = {config.get('batch_size', batch_size)}
pipeline_stages = 1
gradient_accumulation_steps = {config.get('gradient_accumulation_steps', grad_accum)}
gradient_clipping = {config.get('gradient_clipping', 1.0)}
warmup_steps = {config.get('warmup_steps', 100)}

# Eval settings
eval_every_n_epochs = {config.get('eval_every_n_epochs', 1)}
eval_before_first_step = {to_toml_value(config.get('eval_before_first_step', True))}
eval_micro_batch_size_per_gpu = {config.get('eval_micro_batch_size_per_gpu', 1)}
eval_gradient_accumulation_steps = {config.get('eval_gradient_accumulation_steps', 1)}

# Misc settings
save_every_n_epochs = {config.get('save_every_n_epochs', save_every)}
checkpoint_every_n_minutes = {config.get('checkpoint_every_n_minutes', 120)}
activation_checkpointing = {to_toml_value(config.get('activation_checkpointing', True))}
partition_method = {to_toml_value(config.get('partition_method', 'parameters'))}
save_dtype = {to_toml_value(config.get('save_dtype', 'bfloat16'))}
caching_batch_size = {config.get('caching_batch_size', 1)}
steps_per_print = {config.get('steps_per_print', 1)}
video_clip_mode = {to_toml_value(config.get('video_clip_mode', 'single_middle'))}
"""

    # Add any additional custom parameters not in the standard set
    standard_params = {
        # Training settings
        'epochs', 'learning_rate', 'batch_size', 'gradient_accumulation_steps',
        'gradient_clipping', 'warmup_steps',
        # Eval settings
        'eval_every_n_epochs', 'eval_before_first_step',
        'eval_micro_batch_size_per_gpu', 'eval_gradient_accumulation_steps',
        # Misc settings
        'save_every_n_epochs', 'checkpoint_every_n_minutes',
        'activation_checkpointing', 'partition_method', 'save_dtype',
        'caching_batch_size', 'steps_per_print', 'video_clip_mode',
        # Model dtype settings
        'model_dtype', 'transformer_dtype',
        # Adapter settings
        'lora_rank', 'lora_dtype',
        # Optimizer settings
        'optimizer_type', 'optimizer_betas', 'optimizer_weight_decay', 'optimizer_eps',
        # Dataset settings (handled in dataset.toml)
        'frame_buckets', 'resolution'
    }

    custom_params = {k: v for k, v in training.items() if k not in standard_params}
    if custom_params:
        content += "\n# Custom parameters (user-provided)\n"
        for key, value in custom_params.items():
            content += f"{key} = {to_toml_value(value)}\n"

    content += "\n[model]\n"

    # Get dtype settings from config
    model_dtype = config.get('model_dtype', 'bfloat16')
    transformer_dtype = config.get('transformer_dtype')  # None by default

    # Helper to add transformer_dtype line only if set
    def get_transformer_dtype_line(default_value=None):
        """Return transformer_dtype line if configured, or default if provided."""
        dtype_val = transformer_dtype if transformer_dtype else default_value
        if dtype_val:
            return f"transformer_dtype = {to_toml_value(dtype_val)}\n"
        return ""

    # Model-specific configuration
    if model_type == "flux":
        content += f"""type = 'flux'
diffusers_path = '{model_path}'
dtype = {to_toml_value(model_dtype)}
{get_transformer_dtype_line('float8')}flux_shift = true
"""
    elif model_type == "sdxl":
        content += f"""type = 'sdxl'
ckpt_path = '{model_path}'
dtype = {to_toml_value(model_dtype)}
{get_transformer_dtype_line()}"""
    elif model_type in ["wan13", "wan14b_t2v", "wan14b_i2v"]:
        content += f"""type = 'wan'
ckpt_path = '{model_path}'
dtype = {to_toml_value(model_dtype)}
{get_transformer_dtype_line()}timestep_sample_method = 'logit_normal'
"""
    elif model_type == "qwen":
        content += f"""type = 'qwen_image'
diffusers_path = '{model_path}'
dtype = {to_toml_value(model_dtype)}
{get_transformer_dtype_line('float8')}timestep_sample_method = 'logit_normal'
"""
    elif model_type == "z_image_turbo":
        content += f"""type = 'z_image'
diffusion_model = '{model_path}/z_image_turbo_bf16.safetensors'
vae = '{model_path}/ae.safetensors'
text_encoders = [
    {{path = '{model_path}/qwen_3_4b.safetensors', type = 'lumina2'}}
]
merge_adapters = ['{model_path}/zimage_turbo_training_adapter_v2.safetensors']
dtype = {to_toml_value(model_dtype)}
{get_transformer_dtype_line()}"""

    # Adapter configuration
    content += f"""
[adapter]
type = 'lora'
rank = {lora_rank}
dtype = {to_toml_value(config.get('lora_dtype', 'bfloat16'))}

[optimizer]
type = {to_toml_value(config.get('optimizer_type', 'adamw_optimi'))}
lr = {lr}
betas = {to_toml_value(config.get('optimizer_betas', [0.9, 0.99]))}
weight_decay = {config.get('optimizer_weight_decay', 0.01)}
eps = {config.get('optimizer_eps', 1e-8)}
"""

    return content


# ============================================================================
# Model Download
# ============================================================================

def download_model(model_type: str, hf_token: Optional[str], job: Dict) -> bool:
    """Download the required model from HuggingFace."""
    model_config = MODEL_CONFIGS[model_type]
    model_path = os.path.join(WORKING_DIR, model_config["model_path"])

    # Check if model already exists
    if os.path.exists(model_path):
        if os.path.isdir(model_path) and os.listdir(model_path):
            log(f"Model already exists at {model_path}")
            return True
        elif os.path.isfile(model_path):
            log(f"Model file already exists at {model_path}")
            return True

    send_progress(job, f"Downloading {model_config['model_name']} model...")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Build download command
    cmd = ["huggingface-cli", "download", model_config["hf_repo"]]

    if "hf_file" in model_config:
        cmd.append(model_config["hf_file"])

    cmd.extend(["--local-dir", model_path])

    env = {}
    if hf_token:
        env["HF_TOKEN"] = hf_token

    result = run_command(cmd, env=env)

    if result.returncode != 0:
        log(f"Model download failed", "ERROR")
        return False

    return True


# ============================================================================
# Training
# ============================================================================

def run_training(model_type: str, job: Dict, job_id: str) -> Dict:
    """Run the actual training process."""
    model_config = MODEL_CONFIGS[model_type]
    toml_file = f"job_{job_id}.toml"

    send_progress(job, "Starting training...")

    # Build the training command
    cmd = [
        "deepspeed",
        "--num_gpus=1",
        "train.py",
        "--deepspeed",
        "--config", f"examples/{toml_file}"
    ]

    env = {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1"
    }

    # Start training process
    process = subprocess.Popen(
        cmd,
        cwd=DIFFUSION_PIPE_DIR,
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    output_lines = []
    current_epoch = 0
    last_progress_time = time.time()

    # Stream output and send progress updates
    for line in process.stdout:
        output_lines.append(line)
        log(line.rstrip())

        # Parse epoch information from output
        epoch_match = re.search(r'epoch[:\s]+(\d+)', line.lower())
        if epoch_match:
            new_epoch = int(epoch_match.group(1))
            if new_epoch != current_epoch:
                current_epoch = new_epoch
                send_progress(job, f"Training epoch {current_epoch}...")

        # Send periodic progress updates
        if time.time() - last_progress_time > 60:
            send_progress(job, f"Training in progress (epoch {current_epoch})...")
            last_progress_time = time.time()

    process.wait()

    if process.returncode != 0:
        return {
            "status": "failed",
            "error": "Training failed",
            "output": "".join(output_lines[-100:])  # Last 100 lines
        }

    # Find output files
    # diffusion-pipe creates: output_dir/{timestamp}/global_step{N}/
    output_path = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.exists(output_path):
        return {
            "status": "failed",
            "error": "Training completed but output directory not found"
        }

    # Find timestamp directories (format: YYYYMMDD_HH-MM-SS)
    subdirs = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]
    if not subdirs:
        return {
            "status": "failed",
            "error": "Training completed but no output subdirectories found"
        }

    # Get the most recent timestamp directory
    latest_timestamp_dir = max(subdirs)
    timestamp_path = os.path.join(output_path, latest_timestamp_dir)

    # Find epoch directories inside the timestamp directory
    # IMPORTANT: epoch directories contain safetensors (usable for inference)
    # global_step directories are DeepSpeed checkpoints (for resuming training only)
    step_dirs = [d for d in os.listdir(timestamp_path) if d.startswith("epoch")]
    if not step_dirs:
        # Fallback: check for global_step directories (DeepSpeed format - less ideal)
        step_dirs = [d for d in os.listdir(timestamp_path) if d.startswith("global_step")]
        if step_dirs:
            log("Warning: Only DeepSpeed checkpoints found (global_step). These are .pt files, not safetensors.", "WARNING")

    if not step_dirs:
        return {
            "status": "failed",
            "error": "Training completed but no checkpoint directories found"
        }

    # Get the latest step/epoch
    def extract_number(name):
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else 0

    latest_step = max(step_dirs, key=extract_number)
    latest_step_path = os.path.join(timestamp_path, latest_step)

    return {
        "status": "success",
        "output_path": latest_step_path,
        "epochs_completed": len(step_dirs),
        "latest_epoch": latest_step
    }


# ============================================================================
# Output Handling
# ============================================================================

def upload_results(output_path: str, upload_url: str, job: Dict) -> bool:
    """Upload training results to the specified URL."""
    send_progress(job, "Uploading training results...")

    # Create a zip of the output
    zip_path = output_path + ".zip"
    shutil.make_archive(output_path, 'zip', output_path)

    try:
        with open(zip_path, 'rb') as f:
            response = requests.put(upload_url, data=f, timeout=3600)
            response.raise_for_status()

        log("Upload successful")
        return True
    except Exception as e:
        log(f"Upload failed: {e}", "ERROR")
        return False
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


# ============================================================================
# Main Handler
# ============================================================================

def validate_input(input_data: Dict) -> Optional[str]:
    """Validate the input data and return error message if invalid."""
    if "model_type" not in input_data:
        return "Missing required field: model_type"

    model_type = input_data["model_type"]
    if model_type not in MODEL_CONFIGS:
        return f"Invalid model_type: {model_type}. Must be one of: {list(MODEL_CONFIGS.keys())}"

    if "dataset" not in input_data:
        return "Missing required field: dataset"

    dataset = input_data["dataset"]
    if "type" not in dataset:
        return "Missing required field: dataset.type"

    dataset_type = dataset["type"]
    if dataset_type not in ["images", "videos", "both", "precaptioned"]:
        return f"Invalid dataset.type: {dataset_type}"

    # Check for required URLs
    if dataset_type in ["images", "both"]:
        if "images_url" not in dataset:
            return "images_url is required when dataset.type is 'images' or 'both'"

    if dataset_type in ["videos", "both"]:
        if "videos_url" not in dataset:
            return "videos_url is required when dataset.type is 'videos' or 'both'"

    # Check for required API keys
    model_config = MODEL_CONFIGS[model_type]
    api_keys = input_data.get("api_keys", {})

    if model_config["requires_hf_token"]:
        if "huggingface_token" not in api_keys:
            return f"huggingface_token is required for {model_type} model"

    if dataset_type in ["videos", "both"]:
        if "gemini_api_key" not in api_keys:
            return "gemini_api_key is required for video captioning"

    return None


def handler(job: Dict) -> Dict:
    """
    Main handler function for serverless training.

    Expected input format:
    {
        "model_type": "wan14b_t2v",
        "dataset": {
            "type": "videos",
            "videos_url": "https://...",
            "trigger_word": "ohwx person"
        },
        "training": {
            "epochs": 100,
            "learning_rate": 2e-5,
            "lora_rank": 32
        },
        "api_keys": {
            "huggingface_token": "...",
            "gemini_api_key": "..."
        },
        "output": {
            "upload_url": "https://..."
        }
    }
    """
    job_id = job["id"]
    input_data = job.get("input", {})

    log(f"Starting job {job_id}")
    log(f"Input: {json.dumps(input_data, indent=2)}")

    # Initialize progress tracker (will be None if HuggingFace not configured)
    progress_tracker = None

    try:
        # Validate input
        error = validate_input(input_data)
        if error:
            return {"status": "failed", "error": error}

        model_type = input_data["model_type"]
        dataset_config = input_data["dataset"]
        api_keys = input_data.get("api_keys", {})
        output_config = input_data.get("output", {})

        # Initialize HuggingFace progress tracker EARLY if configured
        # This creates the repo and UPDATES folder before training starts
        hf_config = output_config.get("huggingface")
        if hf_config and hf_config.get("token"):
            log("Initializing HuggingFace progress tracker...")
            progress_tracker = init_progress_tracker(hf_config, job_id)
            if progress_tracker:
                progress_tracker.update(
                    "STARTING",
                    f"Initializing {MODEL_CONFIGS[model_type]['model_name']} training job",
                    model_type=model_type,
                    dataset_type=dataset_config.get("type"),
                    training_config=input_data.get("training", {})
                )
            else:
                log("Warning: Failed to initialize progress tracker, continuing without it", "WARNING")

        send_progress(job, f"Initializing {MODEL_CONFIGS[model_type]['model_name']} training job...")

        # Create working directories
        os.makedirs(WORKING_DIR, exist_ok=True)
        os.makedirs(IMAGE_DATASET_DIR, exist_ok=True)
        os.makedirs(VIDEO_DATASET_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(DIFFUSION_PIPE_DIR, "examples"), exist_ok=True)

        # Determine what data types we have
        # For precaptioned, check which URLs were actually provided
        if dataset_config["type"] == "precaptioned":
            has_images = bool(dataset_config.get("images_url"))
            has_videos = bool(dataset_config.get("videos_url"))
        else:
            has_images = dataset_config["type"] in ["images", "both"]
            has_videos = dataset_config["type"] in ["videos", "both"]

        if has_images and "images_url" in dataset_config:
            if progress_tracker:
                progress_tracker.update("DOWNLOADING", "Downloading image dataset", url=dataset_config["images_url"])
            if not download_and_extract_dataset(dataset_config["images_url"], IMAGE_DATASET_DIR, job):
                if progress_tracker:
                    progress_tracker.error("Failed to download image dataset")
                return {"status": "failed", "error": "Failed to download image dataset"}

        if has_videos and "videos_url" in dataset_config:
            if progress_tracker:
                progress_tracker.update("DOWNLOADING", "Downloading video dataset", url=dataset_config["videos_url"])
            if not download_and_extract_dataset(dataset_config["videos_url"], VIDEO_DATASET_DIR, job):
                if progress_tracker:
                    progress_tracker.error("Failed to download video dataset")
                return {"status": "failed", "error": "Failed to download video dataset"}

        # Run captioning if not precaptioned
        if dataset_config["type"] != "precaptioned":
            trigger_word = dataset_config.get("trigger_word")

            if dataset_config["type"] in ["images", "both"]:
                if progress_tracker:
                    progress_tracker.update("CAPTIONING", "Running image captioning with JoyCaption")
                if not run_image_captioning(trigger_word, job):
                    if progress_tracker:
                        progress_tracker.error("Image captioning failed")
                    return {"status": "failed", "error": "Image captioning failed"}

            if dataset_config["type"] in ["videos", "both"]:
                if progress_tracker:
                    progress_tracker.update("CAPTIONING", "Running video captioning with Gemini")
                gemini_key = api_keys.get("gemini_api_key")
                if not run_video_captioning(gemini_key, job):
                    if progress_tracker:
                        progress_tracker.error("Video captioning failed")
                    return {"status": "failed", "error": "Video captioning failed"}

        # Download model
        if progress_tracker:
            progress_tracker.update("MODEL_DOWNLOAD", f"Downloading {MODEL_CONFIGS[model_type]['model_name']} model")
        hf_token = api_keys.get("huggingface_token")
        if not download_model(model_type, hf_token, job):
            if progress_tracker:
                progress_tracker.error("Model download failed")
            return {"status": "failed", "error": "Model download failed"}

        # Generate configuration files
        send_progress(job, "Generating training configuration...")

        dataset_toml = generate_dataset_toml(input_data, has_images, has_videos)
        dataset_toml_path = os.path.join(DIFFUSION_PIPE_DIR, "examples", "dataset.toml")
        with open(dataset_toml_path, 'w') as f:
            f.write(dataset_toml)
        log(f"Generated dataset.toml")

        model_toml = generate_model_toml(input_data, model_type, job_id)
        model_toml_path = os.path.join(DIFFUSION_PIPE_DIR, "examples", f"job_{job_id}.toml")
        with open(model_toml_path, 'w') as f:
            f.write(model_toml)
        log(f"Generated job config: job_{job_id}.toml")

        # Run training
        if progress_tracker:
            progress_tracker.update(
                "TRAINING",
                f"Starting {MODEL_CONFIGS[model_type]['model_name']} LoRA training",
                epochs=input_data.get("training", {}).get("epochs", 100),
                lora_rank=input_data.get("training", {}).get("lora_rank", 32)
            )
        training_result = run_training(model_type, job, job_id)

        if training_result["status"] != "success":
            if progress_tracker:
                progress_tracker.error(f"Training failed: {training_result.get('error', 'Unknown error')}")
            return training_result

        if progress_tracker:
            progress_tracker.update(
                "TRAINING_COMPLETE",
                "Training completed successfully!",
                epochs_completed=training_result["epochs_completed"],
                latest_epoch=training_result["latest_epoch"]
            )

        send_progress(job, "Training completed! Preparing download link...")

        # Process output and generate download link
        if progress_tracker:
            progress_tracker.update("UPLOADING", "Uploading trained LoRA to HuggingFace")
        output_result = upload_and_get_download_url(
            output_dir=training_result["output_path"],
            job_id=job_id,
            output_config=output_config,
            progress_tracker=progress_tracker
        )

        # Build the final response
        output_response = {
            "status": "success",
            "job_id": job_id,
            "model_type": model_type,
            "model_name": MODEL_CONFIGS[model_type]["model_name"],
            "epochs_completed": training_result["epochs_completed"],
            "latest_epoch": training_result["latest_epoch"],
            "output": {
                "download_url": output_result.get("download_url"),
                "safetensors_url": output_result.get("safetensors_url"),  # Direct download link
                "repo_url": output_result.get("repo_url"),  # HuggingFace repo URL
                "download_method": output_result.get("download_method"),
                "expires_in_days": output_result.get("expires_in_days"),
                "archive_size_mb": output_result.get("archive_size_mb"),
                "network_volume_path": output_result.get("network_volume_path"),
            }
        }

        # Add any notes (like one-time download warning for file.io)
        if output_result.get("note"):
            output_response["output"]["note"] = output_result["note"]

        # Add errors if any occurred
        if output_result.get("errors"):
            output_response["output"]["warnings"] = output_result["errors"]

        # Add fallback instructions if no download URL was generated
        if not output_result.get("download_url"):
            output_response["output"]["fallback_instructions"] = output_result.get(
                "fallback_instructions",
                "Files are available on the network volume at the specified path."
            )

        # Log the download URL for easy access
        if output_result.get("download_url"):
            log(f"Download URL generated: {output_result['download_url']}")
            send_progress(job, f"Download ready! URL valid for {output_result.get('expires_in_days', '?')} days.")
        else:
            log("No download URL generated - check output.warnings for details", "WARNING")

        return output_response

    except Exception as e:
        log(f"Unhandled exception: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        # Write error to progress tracker if available
        if progress_tracker:
            progress_tracker.error(f"Unhandled exception: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    log("Starting Diffusion-Pipe Serverless Handler")
    log(f"NETWORK_VOLUME: {NETWORK_VOLUME}")
    log(f"WORKING_DIR: {WORKING_DIR}")

    runpod.serverless.start({
        "handler": handler
    })
