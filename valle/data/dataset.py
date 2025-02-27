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

import torch, re
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
    ) -> None:
        super().__init__()

        self.text_token_collater = text_token_collater
        self.cut_transforms = ifnone(cut_transforms, [])
        self.feature_input_strategy = feature_input_strategy

        if feature_transforms is None:
            feature_transforms = []
        elif not isinstance(feature_transforms, Sequence):
            feature_transforms = [feature_transforms]

        assert all(
            isinstance(transform, Callable) for transform in feature_transforms
        ), "Feature transforms must be Callable"
        self.feature_transforms = feature_transforms

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        validate_for_tts(cuts)
        for transform in self.cut_transforms:
            cuts = transform(cuts)
        audio, audio_lens = None, None

        audio_features, audio_features_lens = self.feature_input_strategy(cuts)

        for transform in self.feature_transforms:
            audio_features = transform(audio_features)
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

            # ✅ Ensure Atypical (source) is always FIRST, followed by Typical (target)
            final_audio_features = torch.cat([audio_features, target_audio_features], dim=1)

            final_audio_features_lens = audio_features_lens + target_audio_features_lens  # Sum lengths

            # Determine max length for padding. Needs to be the two 'same' utterances when added create the longest sequence
            # Otherwise if we add the features first it will take the two independently largest sequences in the src and tgts
            max_length = final_audio_features_lens.max().item()

            # Create a zeroed tensor 
            final_audio_features = torch.zeros((audio_features.shape[0], max_length, audio_features.shape[2]), dtype=audio_features.dtype, device=audio_features.device)

            # Fill in the data correctly
            for i in range(audio_features.shape[0]):
                src_len = audio_features_lens[i].item()
                tgt_len = target_audio_features_lens[i].item()
                
                final_audio_features[i, :src_len] = audio_features[i, :src_len]  # Place source first
                final_audio_features[i, src_len:src_len + tgt_len] = target_audio_features[i, :tgt_len]  # Place target after

        else:
            final_audio_features, final_audio_features_lens = audio_features, audio_features_lens

        text_tokens, text_tokens_lens = self.text_token_collater(
            [cut.supervisions[0].custom["tokens"]["text"] for cut in cuts]
        )
        # print(f"utt_id: {[cut.id for cut in cuts]}\n",
        #     f"text {[cut.supervisions[0].text for cut in cuts]}\n",
        #     f"audio: {audio}\n",
        #     f"audio_lens: {audio_lens}\n",
        #     f"audio_features: {final_audio_features}\n",
        #     f"audio_features_lens: {final_audio_features_lens}\n",
        #     f"text_tokens: {text_tokens}\n",
        #     f"text_tokens_lens: {text_tokens_lens}",)
        return {
            "utt_id": [cut.id for cut in cuts],
            "text": [cut.supervisions[0].text for cut in cuts],
            "audio": audio,
            "audio_lens": audio_lens,
            "audio_features": final_audio_features,
            "audio_features_lens": final_audio_features_lens,
            "text_tokens": text_tokens,
            "text_tokens_lens": text_tokens_lens,
        }


def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."
