# TTS Objective Metrics 🎯

This repository comprises a compilation of the objective metrics used in several text-to-speech (TTS) papers.

## Available Metrics
| Metric | Used In |
| ------ | ------ |
| Voicing Decision Error (VDE) | [E2E-Prosody](https://arxiv.org/pdf/1803.09047.pdf), [Mellotron](https://arxiv.org/abs/1910.11997)|
| Gross Pitch Error (GPE) | [E2E-Prosody](https://arxiv.org/pdf/1803.09047.pdf), [Mellotron](https://arxiv.org/abs/1910.11997)|
| F0 Frame Error (FFE) | [E2E-Prosody](https://arxiv.org/pdf/1803.09047.pdf), [Mellotron](https://arxiv.org/abs/1910.11997)|
| Dynamic Time Warping (DTW) | [FastSpeech2](https://arxiv.org/abs/2006.04558) |
| Mel Spectral Distortion (MSD) | [Wave-Tacotron](https://arxiv.org/abs/2011.03568) |
| Mel Cepstral Distortion (MCD) | [E2E-Prosody](https://arxiv.org/pdf/1803.09047.pdf), [Wave-Tacotron](https://arxiv.org/abs/2011.03568) |
| Statistical Moments (STD, SKEW, KURT) | [FastSpeech2](https://arxiv.org/abs/2006.04558) |

## Available Pitch Computation
| Alogrithm | Proposed In |
| ------ | ------ |
| YIN | [(Cheveigné and Kawahara, 2002)](http://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf) |
| DIO | [(Morise, Kawahara, and Katayose, 2009)](https://www.aes.org/e-lib/browse.cfm?elib=15165)|
| PYIN (Testing) | [(Mauch and Dixon, 2014)](https://ieeexplore.ieee.org/document/6853678) |

## How to Run
First, clone and enter the repo:
```sh
git clone https://github.com/AI-Unicamp/TTS-Objective-Metrics
cd TTS-Objective-Metrics
```

Install dependencies:
```
pip install -r requirements.txt
```

Run
```
python evaluate.py
```
evaluate.py is hardcoded to call driver.txt, driver.txt has one test instance
per row in the following format
source_transcript source_wav_path target_transcript target_wav_path
## Repo Organization
📦TTS Objective Metrics\
 ┣ 📂audio\
 ┃ ┣ 📜helpers.py\
 ┃ ┣ 📜pitch.py\
 ┃ ┣ 📜visuals.py\
 ┣ 📂bin\ # this is legacy code
 ┃ ┣ 📜compute_metrics.py\
 ┣ 📂config\
 ┃ ┣ 📜global_config.py\
 ┣ 📂examples\
 ┣ 📂metrics\
 ┣ ┣ 📂sources\
 ┣ ┣ ┣sources.wav
 ┣ ┣ 📂targets\
 ┣ ┣ ┣targets.wav.wav
 ┃ ┣ 📜dists.py\
 ┃ ┣ 📜DTW.py\
 ┃ ┣ 📜FFE.py\
 ┃ ┣ 📜GPE.py\
 ┃ ┣ 📜helpers.py\
 ┃ ┣ 📜MCD.py\
 ┃ ┣ 📜moments.py\
 ┃ ┣ 📜MSD.py\
 ┃ ┣ 📜VDE.py\
 ┃ ┣ 📜WER.py\
 ┃ ┣ 📜SECS.py
 ┣ 📜README.md
 ┣ 📜evaluate.py
 ┣ 📜driver.txt



## How to Contribute
As the repo is still in its infancy, feel free to either open an issue, discussion or send a pull request, or even contact us by e-mail.

## Authors
- Leonardo B. de M. M. Marques (leonardoboulitreau@gmail.com)
- Lucas Hideki Ueda (lucashueda@gmail.com)

## Github references
- [Coqui-AI](https://github.com/coqui-ai/TTS)
- [Facebook Fairseq](https://github.com/pytorch/fairseq)
- [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)
- [NVIDIA Mellotron](https://github.com/NVIDIA/mellotron/tree/d5362ccae23984f323e3cb024a01ec1de0493aff)
- [MAPS](https://github.com/bastibe/MAPS-Scripts)
- [Yin](https://github.com/patriceguyot/Yin)

All references are listened on top of the used code itself.
