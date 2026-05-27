FROM docker.io/pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y vim wget git libsndfile1 espeak-ng pkg-config libhdf5-dev

# 1. PyTorch pinned first — before anything can pull a newer version
RUN pip install --no-cache-dir \
    "torch==1.13.1+cu116" \
    "torchaudio==0.13.1+cu116" \
    --extra-index-url https://download.pytorch.org/whl/cu116

# 2. Core numeric/template pins
RUN pip install --no-cache-dir \
    "numpy==1.24.3" \
    "markupsafe==2.0.1" \
    "jinja2==3.0.3" \
    "setuptools>=65.0"

# 3. Audio/text deps — none of these require torch>=2.0
RUN pip install --no-cache-dir \
    "torchmetrics==0.11.1" \
    "librosa==0.8.1" \
    "phonemizer==3.2.1" \
    "pypinyin==0.48.0" \
    "matplotlib" \
    "pandas" \
    "h5py==3.8.0" --only-binary=h5py \
    "protobuf==3.20.3" \
    "tensorboard==2.11.2"

# Patch librosa numpy API immediately after install
RUN find /opt/conda/lib/python3.10/site-packages/librosa -name "*.py" -exec sed -i \
    -e 's/np\.complex\b/np.complex128/g' \
    -e 's/np\.float\b/np.float64/g' \
    -e 's/np\.int\b/np.int64/g' \
    {} \;

# 4. lhotse pinned to known working version (git head is unpredictable)
RUN pip uninstall -y lhotse || true && \
    pip install --no-cache-dir "lhotse==1.32.1"

# 5. k2 — built specifically for torch 1.13.1+cu116
RUN pip install --no-cache-dir \
    https://huggingface.co/csukuangfj/k2/resolve/main/cuda/k2-1.23.4.dev20230224+cuda11.6.torch1.13.1-cp310-cp310-linux_x86_64.whl

# 6. icefall — --no-deps prevents requirements.txt from upgrading torch
WORKDIR /workspace
RUN git clone https://github.com/k2-fsa/icefall && \
    cd icefall && \
    pip install --no-deps -r requirements.txt

ENV PYTHONPATH=/workspace/icefall:$PYTHONPATH

# 7. Transformers stack — all pinned to torch 1.13.1-compatible versions
#    accelerate==0.20.3 is the last version that doesn't require torch>=2.0
RUN pip install --no-cache-dir \
    "transformers==4.35.2" \
    "huggingface-hub==0.23.0" \
    "tokenizers==0.15.0" \
    "accelerate==0.20.3" \
    "protobuf==3.20.3" \
    "tensorboard==2.11.2" \
    "encodec" \
    "einops" \
    "omegaconf" \
    "soundfile" \
    "sentencepiece"

# 8. speechbrain — 0.5.16 is the last version compatible with torch 1.13.1
#    speechbrain>=1.0.0 requires torch>=2.0.0 and WILL upgrade torch
RUN pip install --no-cache-dir "speechbrain==0.5.16"

# 9. Remaining deps — whisper/pyworld/Levenshtein don't touch torch
RUN pip install --no-cache-dir --no-build-isolation \
    "openai-whisper==20230314" \
    "Levenshtein==0.27.1" \
    "pyworld==0.3.2"

# 10. VallE — --no-deps prevents setup.py from re-resolving torch
WORKDIR /workspace
RUN git clone -b dev-new https://github.com/SlangLab-NU/VallE.git && \
    cd VallE && \
    pip install -e . --no-deps

RUN cp /opt/conda/lib/libpython3.10.so.1.0 /usr/lib/x86_64-linux-gnu/

# 11. Force final versions — nothing installed after this
RUN pip install --force-reinstall --no-cache-dir \
    "numpy==1.24.3" \
    "markupsafe==2.0.1"

RUN pip install --force-reinstall --no-cache-dir --no-deps \
    "torch==1.13.1+cu116" \
    "torchaudio==0.13.1+cu116" \
    --extra-index-url https://download.pytorch.org/whl/cu116

# 12. Verify torch version — build will fail here if something went wrong
RUN python -c "import torch; assert torch.__version__ == '1.13.1+cu116', f'Wrong torch version: {torch.__version__}'"

ENV PYTHONNOUSERSITE=1

WORKDIR /workspace/VallE