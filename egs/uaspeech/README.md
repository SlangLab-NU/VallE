Training JALL-E (Jordan's Vall-e)

1) Working directory

```
cd /home/data1/vall-e.git/VallE
```

2) Run venv

```
source .venv/bin/activate
```

3) Uaspeech directory

```
cd egs/uaspeech
```

4) Prep Data (IF NEEDED)

step1 prepare dataset
**if prep-tts = 0, we will prep data for many-to-one VC, otherwise it will run base tts**
**control-tts and atypical-tts will use all speakers respectively if 1. This is how you can train just control, just atypical or both**

```
bash prepare.sh --stage -1 --stop-stage 2 --prep-tts 1 --control-tts 1 --atypical-tts 0
```

**if you are using block based train/test/dev splits**

```
bash prepare.sh --stage -1 --stop-stage 2 --prep-tts 0 --control-tts 0 --atypical-tts 0 --block-batching 1
```

5) create export directory

```
exp_dir=whatever_name_you_like
```

6) Change k2 tokens from UASpeech to LibriTTS. 


7) train AR decoder
**If running tts ensure --voice-conversion is false**
***When doing VC finetuning, use the --reset-lr tag to prevent the Eve schedular from reseting the learning rate, which limits lower LRs***

Fine tune parameters. For baseline training see main README

```
python3 bin/trainer.py --max-duration 40 --filter-min-duration 0.0 --filter-max-duration 14 --train-stage 1       --num-buckets 6 --dtype "float16" --save-every-n 1000 --valid-interval 500       --model-name valle --share-embedding true --norm-first true --add-prenet false       --decoder-dim 1024 --nhead 16 --num-decoder-layers 6 --prefix-mode 0       --base-lr 0.00001 --warmup-steps 1 --average-period 0       --num-epochs 20 --start-epoch 2 --start-batch 0 --accumulate-grad-steps 4  --voice-conversion 1  --freeze-lower-layers 5 --reset-lr   --exp-dir ${exp_dir} 
```

**Using wavlm or whisper embeddings**
python3 bin/trainer.py --max-duration 40 --filter-min-duration 0.0 --filter-max-duration 14 --train-stage 1       --num-buckets 6 --dtype "float16" --save-every-n 1000 --valid-interval 500       --model-name valle --share-embedding true --norm-first true --add-prenet false       --decoder-dim 1024 --nhead 16 --num-decoder-layers 6 --prefix-mode 0       --base-lr 0.00001 --warmup-steps 1 --average-period 0       --num-epochs 20 --start-epoch 18 --start-batch 0 --accumulate-grad-steps 4  --voice-conversion 1  --freeze-lower-layers 5 --reset-lr --use-model-embeddings "wavlm" --embed-dim 768  --exp-dir ${exp_dir}


8) copy best valid loss to epoch 2

```
cp ${exp_dir}/best-valid-loss.pt ${exp_dir}/epoch-2.pt
```

9) Train NAR Decoder

```
python3 bin/trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2       --num-buckets 6 --dtype "float32" --save-every-n 1000 --valid-interval 500       --model-name valle --share-embedding true --norm-first true --add-prenet false       --decoder-dim 1024 --nhead 16 --num-decoder-layers 6 --prefix-mode 0       --base-lr 0.00001 --warmup-steps 1 --average-period 0       --num-epochs 20 --start-epoch 2 --start-batch 0 --accumulate-grad-steps 4  --voice-conversion 1  --freeze-lower-layers 5 --reset-lr  --block-batching 1 --exp-dir ${exp_dir}
```

10) Run inference

```
python create_inference_text.py --atyp-speakers "very_low" --exp-dir ${exp_dir}
```

```
python3 bin/infer.py --output-dir infer/demos     --checkpoint=${exp_dir}/best-valid-loss.pt   --text-prompts ""  --audio-prompts "" --text atyp_to_atyp_inference_list.txt
```

Textless


```
python3 bin/infer.py --output-dir infer/demos     --checkpoint=${exp_dir}/best-valid-loss.pt   --textless True  --audio-prompts "" --text atyp_to_atyp_inference_list.txt
```

```
cd VallE/tts-obj-metrics
python evaluate.py
```

```
python3 bin/infer.py --output-dir infer/demos     --checkpoint=${exp_dir}/best-valid-loss.pt     --atypical-audio /home/data1/vall-e.git/VallE/egs/uaspeech/audioSamples/CF02/CF02_B1_C1_M2.wav --text "COMMAND"
```