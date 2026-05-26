FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install numpy > 1.24 as required
RUN pip install "numpy>=1.24.0"

RUN apt-get update && \
    apt-get upgrade -y
RUN apt-get install -y vim wget git libsndfile1 espeak-ng

RUN pip install k2==1.24.4.dev20240223+cuda12.1.torch2.1.0 -f https://k2-fsa.github.io/k2/cuda.html

# Install packages with versions compatible with PyTorch 2.1 and numpy >= 1.24
RUN pip install torchmetrics>=0.11.4 librosa>=0.10.0 phonemizer==3.2.1 pypinyin==0.48.0 lhotse matplotlib h5py

# Install newer torchaudio compatible with PyTorch 2.1
RUN pip install torchaudio>=2.1.0

# Install compatible versions
RUN pip install "markupsafe>=2.1.0" "jinja2>=3.1.0" "transformers>=4.30.0"

# Remove wrong whisper package and install OpenAI Whisper
RUN pip uninstall -y whisper && pip install openai-whisper

RUN pip install speechbrain>=0.5.15

WORKDIR /workspace
RUN git clone https://github.com/k2-fsa/icefall && \
    cd icefall && \
    pip install -r requirements.txt
ENV PYTHONPATH=/workspace/icefall:$PYTHONPATH

RUN cp /opt/conda/lib/libpython3.10.so.1.0 /usr/lib/x86_64-linux-gnu/

# Note: With newer numpy/PyTorch, the librosa fixes may not be needed
# but keeping them for safety
RUN find /opt/conda/lib/python3.10/site-packages/librosa -name "*.py" -exec sed -i \
    -e 's/np\.complex\b/np.complex128/g' \
    -e 's/np\.float\b/np.float64/g' \
    -e 's/np\.int\b/np.int64/g' \
    {} \; 2>/dev/null || true

WORKDIR /workspace
RUN git clone -b sap_data_analysis https://github.com/SlangLab-NU/VallE.git && \
    cd VallE && \
    pip install -e .

WORKDIR /workspace/vall-e