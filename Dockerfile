# Optional cloud GPU environment. Choose an image/tag that matches the host NVIDIA driver.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /workspace/NoBannedSex
COPY pyproject.toml ./
COPY nobannedsex/ nobannedsex/
RUN pip install --no-cache-dir -e '.[tokenizer,export]'
COPY . .
CMD ["bash"]
