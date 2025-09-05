FROM docker.io/pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Pin NumPy to a specific version compatible with PyTorch 1.13.1
RUN pip install "numpy==1.24"

RUN apt-get update && \
    apt-get upgrade -y
RUN apt-get install -y vim wget git libsndfile1 espeak-ng

RUN pip install torchmetrics==0.11.1 librosa==0.8.1 phonemizer==3.2.1 pypinyin==0.48.0 lhotse matplotlib h5py

# Install compatible versions for PyTorch 1.13.1
RUN pip install "markupsafe==2.0.1" "jinja2<3.1.0" "transformers==4.21.3"

RUN pip install https://huggingface.co/csukuangfj/k2/resolve/main/cuda/k2-1.23.4.dev20230224+cuda11.6.torch1.13.1-cp310-cp310-linux_x86_64.whl

RUN pip install torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip uninstall -y whisper && pip install openai-whisper==20230314

RUN pip install speechbrain==1.0.2

RUN pip install Levenshtein==0.27.1

RUN pip install pyworld

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

# Ensure NumPy compatibility after all installations
RUN pip install --force-reinstall "numpy==1.24"

WORKDIR /workspace
RUN git clone -b dev-new https://github.com/SlangLab-NU/VallE.git && \
    cd VallE && \
    pip install -e .

WORKDIR /workspace/vall-e