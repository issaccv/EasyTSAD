FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /workspace

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# 设置默认命令
CMD ["/bin/bash"]