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

import torch
from lhotse import validate
from lhotse.cut import CutSet, MonoCut
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone

from valle.data.collation import TextTokenCollater

class AudioToAudioDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the audio-to-audio task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'utt_id': str
            'audio': (NumSamples,) float tensor
            'audio_lens': int tensor
            'audio_features': (NumFrames x NumFeatures) float tensor
            'audio_features_lens': int tensor
            'target_audio': (NumSamples,) float tensor
            'target_audio_lens': int tensor
            'target_audio_features': (NumFrames x NumFeatures) float tensor
            'target_audio_features_lens': int tensor
        }
    """
    # Removed text collator.
    def __init__(
        self,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        feature_input_strategy: BatchIO = PrecomputedFeatures(),
        feature_transforms: Union[Sequence[Callable], Callable] = None,
    ) -> None:
        super().__init__()

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

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.tensor]:
        validate_for_audio_to_audio(cuts)

        for transform in self.cut_transforms:
            cuts = transform(cuts)

        # print(f"Processing cuts: {[cut.id for cut in cuts]}")

        if False:  # not used
            audio, audio_lens = collate_audio(cuts)
        else:  # for sharing tokenized features in different machines
            audio, audio_lens = None, None

        # audio, audio_lens = collate_audio(cuts)
        audio_features, audio_features_lens = self.feature_input_strategy(cuts)

        for transform in self.feature_transforms:
            audio_features = transform(audio_features)

        # Ensure 'target_recording' is properly converted to a Cut object
        target_audio_cuts = []
        for cut in cuts:
            target_recording = cut.custom['target_recording']
            if isinstance(target_recording, dict):
                target_cut = MonoCut.from_dict(target_recording)
                # print(f"Target MonoCut ID: {target_cut.id}, has_features: {target_cut.has_features}")
                if not target_cut.has_features:
                    print(f"Features missing for target recording: {target_cut}")
                else:
                    target_audio_cuts.append(target_cut)
            else:
                target_audio_cuts.append(target_recording)


        # Convert list of MonoCut to CutSet
        target_cuts = CutSet.from_cuts(target_audio_cuts)
        # print(f"Number of target cuts: {len(target_cuts)}")
            
        # Apply the same transformations to target_cuts
        for transform in self.cut_transforms:
            target_cuts = transform(target_cuts)

        assert all(cut.has_features for cut in target_cuts)

        target_audio, target_audio_lens = None, None
        target_audio_features, target_audio_features_lens = self.feature_input_strategy(target_cuts)

        for transform in self.feature_transforms:
            target_audio_features = transform(target_audio_features)

        # print(f"Processed batch for cuts: {[cut.id for cut in cuts]}")

        # print("GET ITEM WAS CALLED")
        # print(f"utt_id: {[cut.id for cut in cuts]}")
        # print(f"audio: {audio}")
        # print(f"audio_lens: {audio_lens}")
        # print(f"audio_features: {audio_features}")
        # print(f"audio_features_lens: {audio_features_lens}")
        # print(f"target_audio: {target_audio}")
        # print(f"target_audio_lens: {target_audio_lens}")
        # print(f"target_audio_features: {target_audio_features}")
        # print(f"target_audio_features_lens: {target_audio_features_lens}")

        return {
            "utt_id": [cut.id for cut in cuts],
            "audio": audio,
            "audio_lens": audio_lens,
            "audio_features": audio_features,
            "audio_features_lens": audio_features_lens,
            "target_audio": target_audio,
            "target_audio_lens": target_audio_lens,
            "target_audio_features": target_audio_features,
            "target_audio_features_lens": target_audio_features_lens,
        }


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

        if False:  # not used
            audio, audio_lens = collate_audio(cuts)
        else:  # for sharing tokenized features in different machines
            audio, audio_lens = None, None

        audio_features, audio_features_lens = self.feature_input_strategy(cuts)

        for transform in self.feature_transforms:
            audio_features = transform(audio_features)

        text_tokens, text_tokens_lens = self.text_token_collater(
            [cut.supervisions[0].custom["tokens"]["text"] for cut in cuts]
        )
        
        return {
            "utt_id": [cut.id for cut in cuts],
            "text": [cut.supervisions[0].text for cut in cuts],
            "audio": audio,
            "audio_lens": audio_lens,
            "audio_features": audio_features,
            "audio_features_lens": audio_features_lens,
            "text_tokens": text_tokens,
            "text_tokens_lens": text_tokens_lens,
        }


def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."

def validate_for_audio_to_audio(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            'target_recording' in cut.custom
        ), "Each cut must have a 'target_recording' in its custom field."