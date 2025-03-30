import os
import glob
import librosa 

# Load Audios
def read_folder(path):
    out = list()
    for filename in sorted(glob.glob(os.path.join(path, '*.wav'))):
        x, sr = librosa.load(filename)
        out.append(x)
    return out, sr


def load_audio_paths(audio_path_list):
    out = list()
    for filename in audio_path_list:
        x, sr = librosa.load(filename)
        out.append(x)
    return out, sr
