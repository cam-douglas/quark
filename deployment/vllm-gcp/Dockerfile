
FROM vllm/vllm-openai:latest

# Install additional dependencies for Quark integration
RUN pip install google-cloud-storage transformers

# Copy Quark brain wrapper
COPY brain/externals/vllm_brain_wrapper.py /app/vllm_brain_wrapper.py

# Set environment variables
ENV VLLM_CPU_KVCACHE_SPACE=4
ENV HF_TOKEN=${HF_TOKEN}

# Expose port
EXPOSE 8000

# Start vLLM server with brain optimizations
CMD ["python", "-m", "vllm.entrypoints.openai.api_server",      "--model", "/models/gpt2-small",      "--tensor-parallel-size", "1",      "--gpu-memory-utilization", "0.8",      "--max-model-len", "2048",      "--host", "0.0.0.0",      "--port", "8000"]
