FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# # 音声を扱う場合
# RUN apt-get update && \
#     apt-get install -y swig libsndfile1-dev libasound2-dev && \
#     apt-get clean

WORKDIR /app

# install requirements
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /uvx /bin/
COPY pyproject.toml uv.lock .python-version /app/
RUN uv sync --no-install-project
