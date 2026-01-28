FROM nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04

WORKDIR /VLM

# Install Python stack
RUN apt-get update \
    && apt-get --yes --no-install-recommends install \
        python3.10 \
        python3-pip python3-venv python3-wheel python3-setuptools \
        build-essential cmake \
        graphviz git openssh-client \
        libssl-dev libffi-dev \
        curl\
        wget\
    && rm -rf /var/lib/apt/lists/*

COPY . /VLM/
ENV DISABLE_VERSION_CHECK=1
RUN cd /VLM && pip3 install --no-cache-dir -e . \
    && pip3 install --no-cache-dir git+https://github.com/huggingface/transformers\
    && rm -rf ~/.cache/pip
