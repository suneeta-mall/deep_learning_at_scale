FROM python:3.9-slim-bullseye

ARG DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update -yqq && \
    apt-get install --no-install-recommends -yqq \
    software-properties-common \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    unzip \
    gcc \
    wget \
    dirmngr \
    libopenmpi-dev \
    gnupg && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub && \
    add-apt-repository -y "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /" && \
    add-apt-repository -y contrib && \
    apt-get update -yqq && \
    apt-get -yqq install cuda-12.1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /deep_learning_at_scale

# Install pip and build tools
RUN --mount=type=cache,target=/root/.cache \
    python -m pip install -Uq pip build

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache \
    python -m pip install --no-compile -r requirements.txt

# Copy project files and install the project
COPY . ./
RUN python -m pip install --no-deps -qe .