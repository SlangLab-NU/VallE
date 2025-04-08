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

5) create export directory

```
exp_dir=whatever_name_you_like
```

6) train AR decoder
**If running tts ensure --voice-conversion is false**

```
python3 bin/trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1       --num-buckets 6 --dtype "bfloat16" --save-every-n 1000 --valid-interval 500       --model-name valle --share-embedding true --norm-first true --add-prenet false       --decoder-dim 1024 --nhead 4 --num-decoder-layers 4 --prefix-mode 0       --base-lr 0.05 --warmup-steps 200 --average-period 0       --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4  --voice-conversion false     --exp-dir ${exp_dir} 
```

7) copy best valid loss to epoch 2

```
cp ${exp_dir}/best-valid-loss.pt ${exp_dir}/epoch-2.pt
```

8) Train NAR Decoder

```
python3 bin/trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2       --num-buckets 6 --dtype "float32" --save-every-n 1000 --valid-interval 500       --model-name valle --share-embedding true --norm-first true --add-prenet false       --decoder-dim 1024 --nhead 4 --num-decoder-layers 4 --prefix-mode 0       --base-lr 0.05 --warmup-steps 200 --average-period 0       --num-epochs 60 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 8       --exp-dir ${exp_dir}
```

9) Run inference

```
python create_inference_text.py --exp-dir ${exp_dir}
```

```
python3 bin/infer.py --output-dir infer/demos     --checkpoint=${exp_dir}/best-valid-loss.pt   --text-prompts ""  --audio-prompts "" --text atyp_to_atyp_inference_list.txt
```

```
cd VallE/tts-obj-metrics
python evaluate.py
```

```
python3 bin/infer.py --output-dir infer/demos     --checkpoint=${exp_dir}/best-valid-loss.pt     --atypical-audio /home/data1/vall-e.git/VallE/egs/uaspeech/audioSamples/CF02/CF02_B1_C1_M2.wav --text "COMMAND"
```