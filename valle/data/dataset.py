# Copyright      2023                           (authors: Feiteng Li)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
modified from lhoste.dataset.speech_synthesis.py
"""

from typing import Callable, Dict, List, Sequence, Union

import torch, re, whisper, torchaudio
import torch.nn.functional as F
from lhotse import validate
from lhotse.cut import CutSet, MonoCut
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone

from valle.data.collation import TextTokenCollater


class SpeechSynthesisDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech synthesis(e.g. TTS) task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'audio_lens': (B, ) int tensor
            'text': str
            'audio_features': (B x NumFrames x NumFeatures) float tensor
            'audio_features_lens': (B, ) int tensor
            'text_tokens': (B x NumTextTokens) long tensor
            'text_tokens_lens': (B, ) int tensor
        }
    """

    def __init__(
        self,
        text_token_collater: TextTokenCollater,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        feature_input_strategy: BatchIO = PrecomputedFeatures(),
        feature_transforms: Union[Sequence[Callable], Callable] = None,
        # WHISPER FEATURES
        use_whisper_embeddings: bool = True,
        whisper_model_name: str = "tiny",
    ) -> None:
        super().__init__()

        self.text_token_collater = text_token_collater
        self.cut_transforms = ifnone(cut_transforms, [])
        self.feature_input_strategy = feature_input_strategy

        # WHISPER INITIALIZATION
        self.use_whisper_embeddings = use_whisper_embeddings
        if self.use_whisper_embeddings:
            print(f"Loading Whisper {whisper_model_name} for embeddings...")
            # Force to CPU for data loading to avoid device issues
            self.device = "cpu"  # Start with CPU to avoid device conflicts
            self.whisper_model = whisper.load_model(whisper_model_name, device=self.device)
            self.whisper_model.eval()
            print(f"Whisper model loaded on {self.device}")
            print(f"Whisper model actual device: {next(self.whisper_model.parameters()).device}")

        if feature_transforms is None:
            feature_transforms = []
        elif not isinstance(feature_transforms, Sequence):
            feature_transforms = [feature_transforms]

        assert all(
            isinstance(transform, Callable) for transform in feature_transforms
        ), "Feature transforms must be Callable"
        self.feature_transforms = feature_transforms

    def extract_whisper_embeddings(self, audio_tensor: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """
        Extract Whisper embeddings from audio
        """
        with torch.no_grad():
            # Ensure 1D audio
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.squeeze()
            
            # Resample to 16kHz if needed
            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
                audio_tensor = resampler(audio_tensor)
            
            # Move audio to same device as Whisper model
            audio_tensor = audio_tensor.to(self.device)
            
            # Stores original audio length
            original_length = len(audio_tensor)

            # Pad or trim to 30 seconds (Whisper's expected length)  
            target_length = 480000
            if len(audio_tensor) > target_length:
                audio_tensor = audio_tensor[:target_length] 
            else:
                padding = target_length - len(audio_tensor) 
                audio_tensor = F.pad(audio_tensor, (0, padding)) 

            # Whisper preprocessing
            mel = whisper.log_mel_spectrogram(audio_tensor)
            
            # Ensure mel is on correct device
            mel = mel.to(self.device)
            
            # Extract embeddings from Whisper encoder
            embeddings = self.whisper_model.embed_audio(mel.unsqueeze(0))
            
            # CALCULATE HOW MANY EMBEDDING FRAMES CORRESPOND TO ORIGINAL AUDIO
            # Whisper's encoder downsamples by a factor (usually 2x from mel, then more in transformer)
            # For Whisper base: 30 seconds -> 1500 embedding frames
            # So: frames_per_second = 1500 / 30 = 50 frames per second
            frames_per_second = embeddings.shape[1] / 30.0  # 30 seconds total
            original_duration_seconds = original_length / 16000.0  # 16kHz audio
            original_embedding_frames = int(original_duration_seconds * frames_per_second)
            
            # TRIM EMBEDDINGS TO ORIGINAL LENGTH
            embeddings_trimmed = embeddings[:, :original_embedding_frames, :]
            
            # Move embeddings back to CPU to save GPU memory
            embeddings_trimmed = embeddings_trimmed.cpu()

            print(f"Audio: {original_length} samples ({original_duration_seconds:.2f}s) -> "
              f"Embeddings: {original_embedding_frames}/{embeddings.shape[1]} frames")
            
            return embeddings

    # New Whisper Methods
    def collate_whisper_embeddings(self, embeddings_list: List[torch.Tensor]) -> tuple:
        """
        Collate Whisper embeddings like text tokens
        """
        if not embeddings_list:
            return None, None
            
        batch_size = len(embeddings_list)
        max_len = max(emb.shape[1] for emb in embeddings_list)
        embed_dim = embeddings_list[0].shape[-1]
        
        # Create padded tensor
        padded_embeddings = torch.zeros(batch_size, max_len, embed_dim)
        embedding_lens = torch.zeros(batch_size, dtype=torch.long)

        for i, emb in enumerate(embeddings_list):
            seq_len = emb.shape[1]
            padded_embeddings[i, :seq_len] = emb.squeeze(0)
            embedding_lens[i] = seq_len
            
        return padded_embeddings, embedding_lens # (B, T, D), (B,)


    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        
        validate_for_tts(cuts)
        
        for transform in self.cut_transforms:
            cuts = transform(cuts)
        audio, audio_lens = None, None

        source_audio_features, source_audio_features_lens = self.feature_input_strategy(cuts)

        for transform in self.feature_transforms:
            source_audio_features = transform(source_audio_features)
        
        try:
            is_vc_model = "target_recording" in cuts[0].custom if len(cuts) > 0 else False
        except TypeError:
            is_vc_model = False
        
        if is_vc_model:
            # extract target audio
            target_audio_cuts = []
            seen_target_ids = set()

            for cut in cuts:
                target_recording = cut.custom['target_recording']
                if isinstance(target_recording, dict):
                    target_cut = MonoCut.from_dict(target_recording)
                    if not target_cut.has_features:
                        print(f"Features missing for target recording: {target_cut}")
                elif isinstance(target_recording, MonoCut):
                    target_cut = target_recording
                else:
                    raise TypeError(f"Unexpected target_recording type: {type(target_recording)}")
                
                target_cut_id = target_cut.id
                # Check for duplicate
                if target_cut_id in seen_target_ids:
                    counter = 1
                    while f"{target_cut_id}_batch{counter}" in seen_target_ids:
                        counter += 1
                    unique_tgt_id = f"{target_cut_id}_batch{counter}"
                else:
                    unique_tgt_id = target_cut_id

                # Register the unique ID
                seen_target_ids.add(unique_tgt_id)

                # Clone target cut with a guaranteed unique ID
                target_cut = MonoCut(
                    id=unique_tgt_id,
                    start=target_cut.start,
                    duration=target_cut.duration,
                    channel=target_cut.channel,
                    recording=target_cut.recording,
                    features=target_cut.features,
                    supervisions=target_cut.supervisions,
                    custom=target_cut.custom,
                )
                target_audio_cuts.append(target_cut)
                
            if target_audio_cuts:
                # TODO do try except here
                try:
                    target_cuts = CutSet.from_cuts(target_audio_cuts)
                except AssertionError as e:
                    if "Duplicated manifest ID" in str(e):
                        print(f"⚠️ Duplicate target cut detected but continuing: {e}")
                    else:
                        raise e
                    
                target_audio_features, target_audio_features_lens = self.feature_input_strategy(target_cuts)

                # Ensure Atypical (source) is always FIRST, followed by Typical (target)
                concat_audio_features = torch.cat([source_audio_features, target_audio_features], dim=1)

                concat_audio_features_lens = source_audio_features_lens + target_audio_features_lens  # Sum lengths

                # Determine max length for padding. Needs to be the two 'same' utterances when added create the longest sequence
                # Otherwise if we add the features first it will take the two independently largest sequences in the src and tgts
                max_length = concat_audio_features_lens.max().item()

                # Create a zeroed tensor 
                concat_audio_features = torch.zeros((source_audio_features.shape[0], max_length, source_audio_features.shape[2]), dtype=source_audio_features.dtype, device=source_audio_features.device)

                # Fill in the data correctly
                for i in range(source_audio_features.shape[0]):
                    src_len = source_audio_features_lens[i].item()
                    tgt_len = target_audio_features_lens[i].item()
                    
                    concat_audio_features[i, :src_len] = source_audio_features[i, :src_len]  # Place source first
                    concat_audio_features[i, src_len:src_len + tgt_len] = target_audio_features[i, :tgt_len]  # Place target after

            else:
                concat_audio_features, concat_audio_features_lens = source_audio_features, source_audio_features_lens

            # MODIFIED: Handle text tokens vs Whisper embeddings
            if self.use_whisper_embeddings:
                # Extract Whisper embeddings from source audio
                whisper_embeddings_list = []
                for cut in cuts:
                    audio = cut.load_audio()
                    audio_tensor = torch.from_numpy(audio).float()
                    embeddings = self.extract_whisper_embeddings(audio_tensor, cut.sampling_rate)
                    whisper_embeddings_list.append(embeddings)
                
                text_tokens, text_tokens_lens = self.collate_whisper_embeddings(whisper_embeddings_list)
            else:
                # Your original text token extraction (unchanged)
                text_tokens, text_tokens_lens = self.text_token_collater(
                    [cut.supervisions[0].custom["tokens"]["text"] for cut in cuts]
                )
            # print(f"utt_id: {[cut.id for cut in cuts]}\n",
            #     f"text {[cut.supervisions[0].text for cut in cuts]}\n",
            #     f"audio: {audio}\n",
            #     f"audio_lens: {audio_lens}\n", 
            #     f"atypical_audio_features: {source_audio_features}\n",
            #     f"atypical_audio_lens: {source_audio_features_lens}\n",
            #     f"audio_features: {target_audio_features}\n",
            #     f"audio_features_lens: {target_audio_features_lens}\n",
            #     f"text_tokens: {text_tokens}\n",
            #     f"text_tokens_lens: {text_tokens_lens}",)
            return {
                "utt_id": [cut.id for cut in cuts],
                "text": [cut.supervisions[0].text for cut in cuts],
                "audio": audio,
                "audio_lens": audio_lens,
                "atypical_audio_features": source_audio_features,
                "atypical_audio_features_lens": source_audio_features_lens,
                "target_audio_features": target_audio_features,
                "target_audio_features_lens": target_audio_features_lens,
                "text_tokens": text_tokens,
                "text_tokens_lens": text_tokens_lens,
            }
        
        else:
            text_tokens, text_tokens_lens = self.text_token_collater(
                [cut.supervisions[0].custom["tokens"]["text"] for cut in cuts]
            )
            return {
            "utt_id": [cut.id for cut in cuts],
            "text": [cut.supervisions[0].text for cut in cuts],
            "audio": audio,
            "audio_lens": audio_lens,
            "target_audio_features": source_audio_features,
            "target_audio_features_lens": source_audio_features_lens,
            "text_tokens": text_tokens,
            "text_tokens_lens": text_tokens_lens,
            }


def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."
