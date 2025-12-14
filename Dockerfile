# Dockerfile for Diffusion-Pipe Serverless Endpoint
# Based on hearmeman/diffusion-pipe:v11

FROM hearmeman/diffusion-pipe:v11

LABEL maintainer="mirzabicer"
LABEL description="Serverless version of Diffusion-Pipe for LoRA training"
LABEL version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Create serverless directory
RUN mkdir -p /serverless

# Install RunPod SDK and other required packages
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    requests \
    huggingface-hub[cli]

# Upgrade key packages
RUN pip install --upgrade \
    transformers \
    "peft>=0.17.0" \
    accelerate

# Clone the Hearmeman scripts repository for captioning
# We only need the Captioning folder
RUN git clone --depth 1 https://github.com/Hearmeman24/runpod-diffusion_pipe.git /tmp/hearmeman_scripts && \
    cp -r /tmp/hearmeman_scripts/Captioning /serverless/ && \
    chmod +x /serverless/Captioning/JoyCaption/JoyCaptionRunner.sh 2>/dev/null || true && \
    chmod +x /serverless/Captioning/video_captioner.sh 2>/dev/null || true && \
    rm -rf /tmp/hearmeman_scripts

# Copy our serverless handler, output handler, and startup script
COPY handler.py /serverless/handler.py
COPY output_handler.py /serverless/output_handler.py
COPY start_serverless.sh /serverless/start_serverless.sh

# Make scripts executable
RUN chmod +x /serverless/start_serverless.sh && \
    chmod +x /serverless/handler.py

# Create test input file for local testing (optional)
COPY test_input.json /serverless/test_input.json

# Remove the original start script (pod-specific)
RUN rm -f /start_script.sh

# Set the entrypoint to our serverless startup script
CMD ["/serverless/start_serverless.sh"]

# Health check (optional but useful)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import runpod" || exit 1
