"""
Output Handler Module for Diffusion-Pipe Serverless

This module handles uploading training results and generating download links.
Supports multiple strategies:
1. HuggingFace Hub (preferred - uploads raw .safetensors files)
2. User's own S3-compatible storage (AWS S3, Cloudflare R2, etc.)
3. Free file sharing services (transfer.sh, file.io)
4. Network volume (always as backup)

Also provides HuggingFace-based progress tracking that persists beyond RunPod's job retention.
"""

import os
import json
import shutil
import subprocess
import requests
import tempfile
import glob
import io
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime, timezone
import logging

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# HuggingFace Progress Tracker
# =============================================================================

class HuggingFaceProgressTracker:
    """
    Tracks training progress by writing updates to a HuggingFace repository.

    This provides persistent progress tracking that survives RunPod job purging.
    Updates are written as timestamped text files in an UPDATES/ folder.
    """

    def __init__(self, hf_config: Dict, job_id: str):
        """
        Initialize the progress tracker.

        Args:
            hf_config: HuggingFace configuration with token, repo_id, private
            job_id: Unique job identifier
        """
        self.hf_config = hf_config
        self.job_id = job_id
        self.repo_id = None
        self.api = None
        self.initialized = False
        self.update_count = 0

    def initialize(self) -> bool:
        """
        Initialize the tracker by creating the repo and UPDATES folder.

        Returns:
            True if successful, False otherwise
        """
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            logger.error("huggingface_hub not installed")
            return False

        token = self.hf_config.get('token')
        if not token:
            logger.error("HuggingFace token not provided")
            return False

        try:
            self.api = HfApi(token=token)

            # Get or create repo ID
            self.repo_id = self.hf_config.get('repo_id')
            if not self.repo_id:
                try:
                    user_info = self.api.whoami()
                    username = user_info.get('name', 'user')
                except Exception:
                    username = 'user'
                self.repo_id = f"{username}/diffusion-pipe-lora-{self.job_id[:8]}"

            is_private = self.hf_config.get('private', True)

            # Create repo
            logger.info(f"Creating HuggingFace repo for progress tracking: {self.repo_id}")
            create_repo(
                repo_id=self.repo_id,
                token=token,
                private=is_private,
                exist_ok=True,
                repo_type="model"
            )

            # Write initial status file
            self._write_update("INITIALIZED", {
                "job_id": self.job_id,
                "repo_id": self.repo_id,
                "message": "Training job initialized. Progress updates will appear in the UPDATES/ folder.",
                "note": "The final update will contain the download link for your trained LoRA."
            })

            self.initialized = True
            logger.info(f"Progress tracker initialized: {self.repo_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize progress tracker: {e}")
            return False

    def _write_update(self, status: str, data: Dict[str, Any]) -> bool:
        """Write an update file to the UPDATES folder as JSON."""
        if not self.api or not self.repo_id:
            return False

        try:
            import json

            self.update_count += 1
            timestamp = datetime.now(timezone.utc)
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"UPDATES/{self.update_count:04d}_{timestamp_str}_{status}.json"

            # Build structured JSON content
            update_data = {
                "status": status,
                "timestamp": timestamp.isoformat(),
                "job_id": self.job_id,
                "update_number": self.update_count,
                "repo_id": self.repo_id,
                "data": data
            }

            content = json.dumps(update_data, indent=2, default=str)

            # Upload to HuggingFace
            self.api.upload_file(
                path_or_fileobj=content.encode('utf-8'),
                path_in_repo=filename,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Update #{self.update_count}: {status}"
            )

            logger.info(f"Progress update written: {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to write progress update: {e}")
            return False

    def update(self, status: str, message: str, **kwargs) -> bool:
        """
        Write a progress update.

        Args:
            status: Status string (e.g., "TRAINING", "UPLOADING", "COMPLETE")
            message: Human-readable message
            **kwargs: Additional data to include
        """
        if not self.initialized:
            return False

        data = {"message": message}
        data.update(kwargs)
        return self._write_update(status, data)

    def complete(self, download_url: str, safetensors_url: str, **kwargs) -> bool:
        """
        Write the final completion update with download links.

        Args:
            download_url: URL to the HuggingFace repo
            safetensors_url: Direct download URL for the safetensors file
            **kwargs: Additional completion data
        """
        if not self.initialized:
            return False

        data = {
            "message": "Training completed successfully!",
            "repository_url": download_url,
            "direct_download_url": safetensors_url,
            "instructions": "Use the direct_download_url to download your trained LoRA safetensors file.",
        }
        data.update(kwargs)
        return self._write_update("COMPLETE", data)

    def error(self, error_message: str, **kwargs) -> bool:
        """Write an error update."""
        if not self.initialized:
            return False

        data = {"error": error_message}
        data.update(kwargs)
        return self._write_update("ERROR", data)

    def get_repo_id(self) -> Optional[str]:
        """Get the repository ID."""
        return self.repo_id


# Global progress tracker instance (set by handler)
_progress_tracker: Optional[HuggingFaceProgressTracker] = None

def init_progress_tracker(hf_config: Dict, job_id: str) -> Optional[HuggingFaceProgressTracker]:
    """Initialize the global progress tracker."""
    global _progress_tracker
    tracker = HuggingFaceProgressTracker(hf_config, job_id)
    if tracker.initialize():
        _progress_tracker = tracker
        return tracker
    return None

def get_progress_tracker() -> Optional[HuggingFaceProgressTracker]:
    """Get the global progress tracker."""
    return _progress_tracker


class OutputHandler:
    """Handles uploading training results and generating download links."""

    def __init__(self):
        self.max_file_size_mb = 10 * 1024  # 10GB max for transfer.sh
        self.transfer_sh_url = "https://transfer.sh"
        self.file_retention_days = 14

    def create_output_archive(self, output_dir: str, job_id: str) -> Tuple[str, int]:
        """
        Create a zip archive of the training output.

        Args:
            output_dir: Path to the training output directory
            job_id: Unique job identifier

        Returns:
            Tuple of (archive_path, size_in_bytes)
        """
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        # Create archive in temp directory
        archive_name = f"lora_{job_id}"
        archive_base = os.path.join(tempfile.gettempdir(), archive_name)

        logger.info(f"Creating archive from {output_dir}")
        archive_path = shutil.make_archive(
            archive_base,
            'zip',
            root_dir=os.path.dirname(output_dir),
            base_dir=os.path.basename(output_dir)
        )

        size = os.path.getsize(archive_path)
        logger.info(f"Archive created: {archive_path} ({size / 1024 / 1024:.2f} MB)")

        return archive_path, size

    def upload_to_user_s3(
        self,
        file_path: str,
        s3_config: Dict
    ) -> Optional[str]:
        """
        Upload file to user's S3-compatible storage and return presigned download URL.

        Args:
            file_path: Path to the file to upload
            s3_config: Dictionary with S3 configuration:
                - endpoint_url: S3 endpoint (optional for AWS)
                - bucket: Bucket name
                - region: AWS region
                - access_key: Access key ID
                - secret_key: Secret access key
                - key_prefix: Optional prefix for the object key

        Returns:
            Presigned download URL or None if failed
        """
        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import ClientError
        except ImportError:
            logger.error("boto3 not installed. Cannot upload to S3.")
            return None

        try:
            # Configure S3 client
            config = Config(signature_version='s3v4')

            client_kwargs = {
                'aws_access_key_id': s3_config['access_key'],
                'aws_secret_access_key': s3_config['secret_key'],
                'config': config,
            }

            if s3_config.get('endpoint_url'):
                client_kwargs['endpoint_url'] = s3_config['endpoint_url']

            if s3_config.get('region'):
                client_kwargs['region_name'] = s3_config['region']

            s3_client = boto3.client('s3', **client_kwargs)

            # Determine object key
            filename = os.path.basename(file_path)
            key_prefix = s3_config.get('key_prefix', 'diffusion-pipe-outputs')
            object_key = f"{key_prefix}/{filename}"

            # Upload file
            logger.info(f"Uploading to S3: s3://{s3_config['bucket']}/{object_key}")
            s3_client.upload_file(
                file_path,
                s3_config['bucket'],
                object_key,
                ExtraArgs={'ContentType': 'application/zip'}
            )

            # Generate presigned URL (valid for 7 days)
            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': s3_config['bucket'],
                    'Key': object_key
                },
                ExpiresIn=7 * 24 * 3600  # 7 days
            )

            logger.info("S3 upload successful, presigned URL generated")
            return presigned_url

        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {e}")
            return None

    def upload_to_transfer_sh(self, file_path: str) -> Optional[str]:
        """
        Upload file to transfer.sh and return download URL.

        transfer.sh features:
        - Max file size: 10GB
        - Retention: 14 days
        - No registration required

        Args:
            file_path: Path to the file to upload

        Returns:
            Download URL or None if failed
        """
        file_size = os.path.getsize(file_path)
        max_size = self.max_file_size_mb * 1024 * 1024

        if file_size > max_size:
            logger.error(f"File too large for transfer.sh: {file_size / 1024 / 1024:.2f} MB")
            return None

        filename = os.path.basename(file_path)
        upload_url = f"{self.transfer_sh_url}/{filename}"

        try:
            logger.info(f"Uploading to transfer.sh: {filename} ({file_size / 1024 / 1024:.2f} MB)")

            with open(file_path, 'rb') as f:
                response = requests.put(
                    upload_url,
                    data=f,
                    headers={
                        'Max-Days': str(self.file_retention_days),
                    },
                    timeout=3600  # 1 hour timeout for large files
                )

            if response.status_code == 200:
                download_url = response.text.strip()
                logger.info(f"transfer.sh upload successful: {download_url}")
                return download_url
            else:
                logger.error(f"transfer.sh upload failed: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error("transfer.sh upload timed out")
            return None
        except Exception as e:
            logger.error(f"transfer.sh upload error: {e}")
            return None

    def upload_to_fileio(self, file_path: str) -> Optional[str]:
        """
        Upload file to file.io and return download URL.

        file.io features:
        - Simpler API
        - File deleted after first download (one-time link)
        - Max 2GB file size

        Args:
            file_path: Path to the file to upload

        Returns:
            Download URL or None if failed
        """
        file_size = os.path.getsize(file_path)
        max_size = 2 * 1024 * 1024 * 1024  # 2GB

        if file_size > max_size:
            logger.error(f"File too large for file.io: {file_size / 1024 / 1024:.2f} MB")
            return None

        try:
            logger.info(f"Uploading to file.io: {os.path.basename(file_path)}")

            with open(file_path, 'rb') as f:
                response = requests.post(
                    'https://file.io',
                    files={'file': f},
                    data={'expires': '14d'},  # 14 days expiry
                    timeout=3600
                )

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    download_url = result.get('link')
                    logger.info(f"file.io upload successful: {download_url}")
                    return download_url

            logger.error(f"file.io upload failed: {response.text}")
            return None

        except Exception as e:
            logger.error(f"file.io upload error: {e}")
            return None

    def upload_to_huggingface(
        self,
        output_dir: str,
        job_id: str,
        hf_config: Dict,
        progress_tracker: Optional[HuggingFaceProgressTracker] = None
    ) -> Optional[Dict[str, str]]:
        """
        Upload training output files directly to HuggingFace Hub.

        This uploads raw files (not zipped) - ideal for LoRA weights.

        Args:
            output_dir: Path to the training output directory
            job_id: Unique job identifier
            hf_config: Dictionary with HuggingFace configuration:
                - token: HuggingFace token with write access
                - repo_id: Repository ID (e.g., "username/my-lora") - optional, auto-generated if not provided
                - private: Whether the repo should be private (default: True)
            progress_tracker: Optional progress tracker (repo already created)

        Returns:
            Dict with repo_url and safetensors_url, or None if failed
        """
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            logger.error("huggingface_hub not installed. Cannot upload to HuggingFace.")
            return None

        token = hf_config.get('token')
        if not token:
            logger.error("HuggingFace token not provided")
            return None

        try:
            # Use existing API from progress tracker if available
            if progress_tracker and progress_tracker.api:
                api = progress_tracker.api
                repo_id = progress_tracker.repo_id
            else:
                api = HfApi(token=token)

                # Get or create repo ID
                repo_id = hf_config.get('repo_id')
                if not repo_id:
                    try:
                        user_info = api.whoami()
                        username = user_info.get('name', 'user')
                    except Exception:
                        username = 'user'
                    repo_id = f"{username}/diffusion-pipe-lora-{job_id[:8]}"

                is_private = hf_config.get('private', True)

                # Create repo if it doesn't exist
                logger.info(f"Creating/accessing HuggingFace repo: {repo_id}")
                try:
                    create_repo(
                        repo_id=repo_id,
                        token=token,
                        private=is_private,
                        exist_ok=True,
                        repo_type="model"
                    )
                except Exception as e:
                    logger.warning(f"Repo creation warning (may already exist): {e}")

            # Find safetensors files in the output directory
            safetensors_files = []
            all_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, output_dir)
                    all_files.append(rel_path)
                    if file.endswith('.safetensors'):
                        safetensors_files.append(rel_path)

            logger.info(f"Found {len(all_files)} files, {len(safetensors_files)} safetensors")

            # Upload the folder contents
            logger.info(f"Uploading training output to HuggingFace: {repo_id}")
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload LoRA weights from diffusion-pipe training (job: {job_id})"
            )

            # Build URLs
            repo_url = f"https://huggingface.co/{repo_id}"

            # Find the main safetensors file for direct download
            # Priority: adapter_model.safetensors > any other .safetensors
            safetensors_url = None
            if safetensors_files:
                # Look for adapter_model.safetensors first (PEFT standard)
                for sf in safetensors_files:
                    if sf == "adapter_model.safetensors" or sf.endswith("/adapter_model.safetensors"):
                        safetensors_url = f"https://huggingface.co/{repo_id}/resolve/main/{sf}?download=true"
                        break
                # If not found, use the first safetensors file
                if not safetensors_url:
                    safetensors_url = f"https://huggingface.co/{repo_id}/resolve/main/{safetensors_files[0]}?download=true"
            else:
                # No safetensors found - might be .pt files (DeepSpeed format)
                logger.warning("No .safetensors files found in output directory")
                safetensors_url = repo_url  # Fall back to repo URL

            logger.info(f"HuggingFace upload successful: {repo_url}")
            logger.info(f"Direct safetensors URL: {safetensors_url}")

            # Write completion update if progress tracker is active
            if progress_tracker:
                progress_tracker.complete(
                    download_url=repo_url,
                    safetensors_url=safetensors_url,
                    files_uploaded=all_files,
                    safetensors_files=safetensors_files
                )

            return {
                "repo_url": repo_url,
                "safetensors_url": safetensors_url,
                "files": all_files,
                "safetensors_files": safetensors_files
            }

        except Exception as e:
            logger.error(f"HuggingFace upload error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Write error to progress tracker
            if progress_tracker:
                progress_tracker.error(str(e))
            return None

    def upload_to_litterbox(self, file_path: str) -> Optional[str]:
        """
        Upload file to litterbox.catbox.moe and return download URL.

        litterbox features:
        - Free, no registration
        - Up to 1GB files
        - Configurable retention (1h, 12h, 24h, 72h)

        Args:
            file_path: Path to the file to upload

        Returns:
            Download URL or None if failed
        """
        file_size = os.path.getsize(file_path)
        max_size = 1 * 1024 * 1024 * 1024  # 1GB

        if file_size > max_size:
            logger.error(f"File too large for litterbox: {file_size / 1024 / 1024:.2f} MB (max 1GB)")
            return None

        try:
            logger.info(f"Uploading to litterbox: {os.path.basename(file_path)} ({file_size / 1024 / 1024:.2f} MB)")

            with open(file_path, 'rb') as f:
                response = requests.post(
                    'https://litterbox.catbox.moe/resources/internals/api.php',
                    files={'fileToUpload': f},
                    data={
                        'reqtype': 'fileupload',
                        'time': '72h'  # 72 hours retention
                    },
                    timeout=3600
                )

            if response.status_code == 200 and response.text.startswith('https://'):
                download_url = response.text.strip()
                logger.info(f"litterbox upload successful: {download_url}")
                return download_url
            else:
                logger.error(f"litterbox upload failed: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"litterbox upload error: {e}")
            return None

    def process_output(
        self,
        output_dir: str,
        job_id: str,
        output_config: Dict,
        progress_tracker: Optional[HuggingFaceProgressTracker] = None
    ) -> Dict:
        """
        Process training output: upload and return download info.

        Priority order:
        1. HuggingFace (if configured) - uploads raw files, no archive
        2. S3 (if configured) - uploads zip archive
        3. litterbox - uploads zip archive (72h retention)
        4. transfer.sh - uploads zip archive (14 day retention)
        5. file.io - uploads zip archive (one-time download)

        Args:
            output_dir: Path to training output directory
            job_id: Unique job identifier
            output_config: Output configuration from job input:
                - method: Preferred upload method ('huggingface', 's3', 'litterbox', 'transfer_sh', 'auto')
                - huggingface: HuggingFace configuration (token, repo_id, private)
                - s3: S3 configuration for user's bucket
            progress_tracker: Optional HuggingFace progress tracker

        Returns:
            Dictionary with output information and download URLs
        """
        result = {
            "network_volume_path": output_dir,
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "download_url": None,
            "safetensors_url": None,  # Direct download link for safetensors
            "download_method": None,
            "expires_in_days": None,
            "errors": []
        }

        # Determine upload method
        method = output_config.get('method', 'auto')
        hf_config = output_config.get('huggingface')
        s3_config = output_config.get('s3')

        download_url = None
        archive_path = None
        archive_size = 0

        try:
            # ===== HUGGINGFACE (Priority 1) =====
            # HuggingFace uploads raw files directly, no archive needed
            if method in ('huggingface', 'auto') and hf_config:
                logger.info("Attempting HuggingFace upload (raw files)...")
                hf_result = self.upload_to_huggingface(output_dir, job_id, hf_config, progress_tracker)
                if hf_result:
                    result["download_method"] = "huggingface"
                    result["expires_in_days"] = None  # Permanent
                    result["download_url"] = hf_result["safetensors_url"]  # Direct download link
                    result["safetensors_url"] = hf_result["safetensors_url"]
                    result["repo_url"] = hf_result["repo_url"]
                    result["files_uploaded"] = hf_result.get("files", [])
                    result["note"] = "Direct download link for your trained LoRA safetensors file"
                    return result
                elif method == 'huggingface':
                    result["errors"].append("HuggingFace upload failed")

            # For all other methods, we need to create an archive first
            if not download_url:
                try:
                    archive_path, archive_size = self.create_output_archive(output_dir, job_id)
                    result["archive_size_mb"] = round(archive_size / 1024 / 1024, 2)
                except Exception as e:
                    result["errors"].append(f"Failed to create archive: {e}")
                    return result

            # ===== S3 (Priority 2) =====
            if not download_url and method in ('s3', 'auto') and s3_config:
                logger.info("Attempting S3 upload...")
                download_url = self.upload_to_user_s3(archive_path, s3_config)
                if download_url:
                    result["download_method"] = "s3"
                    result["expires_in_days"] = 7
                elif method == 's3':
                    result["errors"].append("S3 upload failed")

            # ===== LITTERBOX (Priority 3) =====
            # litterbox is more reliable than transfer.sh based on user testing
            if not download_url and method in ('litterbox', 'auto'):
                if archive_size <= 1 * 1024 * 1024 * 1024:  # 1GB max
                    logger.info("Attempting litterbox upload...")
                    download_url = self.upload_to_litterbox(archive_path)
                    if download_url:
                        result["download_method"] = "litterbox"
                        result["expires_in_days"] = 3  # 72 hours
                    elif method == 'litterbox':
                        result["errors"].append("litterbox upload failed")

            # ===== TRANSFER.SH (Priority 4) =====
            if not download_url and method in ('transfer_sh', 'auto'):
                logger.info("Attempting transfer.sh upload...")
                download_url = self.upload_to_transfer_sh(archive_path)
                if download_url:
                    result["download_method"] = "transfer_sh"
                    result["expires_in_days"] = self.file_retention_days
                elif method == 'transfer_sh':
                    result["errors"].append("transfer.sh upload failed")

            # ===== FILE.IO (Priority 5 - Last Resort) =====
            if not download_url and method == 'auto':
                if archive_size <= 2 * 1024 * 1024 * 1024:  # 2GB
                    logger.info("Attempting file.io upload...")
                    download_url = self.upload_to_fileio(archive_path)
                    if download_url:
                        result["download_method"] = "file_io"
                        result["expires_in_days"] = 14
                        result["note"] = "One-time download link - file deleted after first download"

            result["download_url"] = download_url

            if not download_url:
                result["errors"].append(
                    "All upload methods failed. Files are still available on network volume."
                )
                result["fallback_instructions"] = (
                    "To retrieve files: 1) Launch a pod with the same network volume, "
                    "2) Navigate to the output path, 3) Download via SCP or web terminal"
                )

        finally:
            # Clean up temporary archive if created
            if archive_path and os.path.exists(archive_path):
                os.remove(archive_path)
                logger.info(f"Cleaned up temporary archive: {archive_path}")

        return result


# Convenience function for use in handler
def upload_and_get_download_url(
    output_dir: str,
    job_id: str,
    output_config: Dict,
    progress_tracker: Optional[HuggingFaceProgressTracker] = None
) -> Dict:
    """
    Convenience wrapper for OutputHandler.process_output()

    Args:
        output_dir: Path to training output directory
        job_id: Unique job identifier
        output_config: Output configuration
        progress_tracker: Optional HuggingFace progress tracker

    Returns:
        Dictionary with output information and download URLs
    """
    handler = OutputHandler()
    return handler.process_output(output_dir, job_id, output_config, progress_tracker)
