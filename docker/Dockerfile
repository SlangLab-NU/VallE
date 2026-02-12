FROM docker.io/pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y vim wget git libsndfile1 espeak-ng && \
    rm -rf /var/lib/apt/lists/*

# Pin critical dependencies FIRST
RUN pip install --no-cache-dir \
    "numpy==1.24.3" \
    "markupsafe==2.0.1" \
    "jinja2==3.0.3"

RUN pip install --no-cache-dir \
    torchmetrics==0.11.1 \
    librosa==0.8.1 \
    phonemizer==3.2.1 \
    pypinyin==0.48.0 \
    lhotse \
    matplotlib \
    h5py \
    pandas

RUN pip install --no-cache-dir \
    "transformers==4.21.3" \
    "tokenizers==0.12.1" \
    "huggingface-hub==0.8.1"

RUN pip install --no-cache-dir \
    https://huggingface.co/csukuangfj/k2/resolve/main/cuda/k2-1.23.4.dev20230224+cuda11.6.torch1.13.1-cp310-cp310-linux_x86_64.whl

RUN pip install --no-cache-dir \
    torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# RUN pip uninstall -y whisper && \
#     pip install --no-cache-dir openai-whisper==20230314

RUN pip install --no-cache-dir speechbrain==1.0.2

RUN pip install --no-cache-dir Levenshtein==0.27.1

RUN pip install --no-cache-dir pyworld

# Install wandb early
RUN pip install --no-cache-dir "wandb==0.18.7"

WORKDIR /workspace
RUN git clone https://github.com/k2-fsa/icefall && \
    cd icefall && \
    pip install -r requirements.txt

ENV PYTHONPATH=/workspace/icefall:$PYTHONPATH

RUN cp /opt/conda/lib/libpython3.10.so.1.0 /usr/lib/x86_64-linux-gnu/

RUN find /opt/conda/lib/python3.10/site-packages/librosa -name "*.py" -exec sed -i \
    -e 's/np\.complex\b/np.complex128/g' \
    -e 's/np\.float\b/np.float64/g' \
    -e 's/np\.int\b/np.int64/g' \
    {} \;

WORKDIR /workspace
RUN git clone -b dev-new https://github.com/SlangLab-NU/VallE.git && \
    cd VallE && \
    pip install -e .

# Force numpy back to compatible version after all installations
RUN pip install --force-reinstall --no-cache-dir "numpy==1.24.3" "markupsafe==2.0.1"

# Install jiwer
RUN pip install --no-cache-dir jiwer

# Install pyarrow with a version that has pre-built wheels (avoid cmake)
RUN pip install --no-cache-dir --only-binary=:all: "pyarrow==14.0.1"

# Install NeMo with constraints
RUN pip install --no-cache-dir --prefer-binary \
    "nemo_toolkit[asr]" \
    "wandb<0.19" \
    "pyarrow==14.0.1"

WORKDIR /workspace/vall-e