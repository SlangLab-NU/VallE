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
from collections import defaultdict
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

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.tensor]:
        for cut in cuts:
            if cut.custom == None or cut.custom == {}:
                print(f"FOUND EMPTY CUT: {cut}")
        cuts = [cut for cut in cuts if cut.custom is not None and cut.custom != {}]
        cuts = CutSet.from_cuts(cuts)
        validate_for_audio_to_audio(cuts)

        for transform in self.cut_transforms:
            cuts = transform(cuts)

        audio, audio_lens = None, None

        audio_features, audio_features_lens = self.feature_input_strategy(cuts)

        for transform in self.feature_transforms:
            audio_features = transform(audio_features)

        # Ensure 'target_recording' is properly converted to a Cut object
        target_audio_cuts = []
        target_counts = defaultdict(int)  # Track occurrences of each target cut
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
                print(f"Unique ID collision! Generating new ID: {unique_tgt_id}")
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
        
        cut_ids = [cut.id for cut in target_audio_cuts]
        duplicate_ids = set([x for x in cut_ids if cut_ids.count(x) > 1])

        if duplicate_ids:
            print(f"⚠️ Duplicate target IDs before creating CutSet: {duplicate_ids}")

        # Convert list of MonoCut to CutSet
        target_cuts = CutSet.from_cuts(target_audio_cuts)
            
        # Apply the same transformations to target_cuts
        for transform in self.cut_transforms:
            target_cuts = transform(target_cuts)

        assert all(cut.has_features for cut in target_cuts)

        target_audio, target_audio_lens = None, None
        target_audio_features, target_audio_features_lens = self.feature_input_strategy(target_cuts)

        for transform in self.feature_transforms:
            target_audio_features = transform(target_audio_features)
        
        text_tokens, text_tokens_lens = self.text_token_collater(
            [cut.supervisions[0].custom["tokens"]["text"] for cut in cuts]
        )
        # print(text_tokens, text_tokens_lens)

        # print(f"Processed batch for cuts: {[cut.id for cut in cuts]}")

        # print("GET ITEM WAS CALLED")
        # print(f"utt_id: {[cut.id for cut in cuts]}")
        # print(f"text: {[cut.supervisions[0].text for cut in cuts]}")
        # print(f"audio: {audio}")
        # print(f"audio_lens: {audio_lens}")
        # print(f"audio_features: {audio_features}")
        # print(f"audio_features_lens: {audio_features_lens}")
        # print(f"target_audio: {target_audio}")
        # print(f"target_audio_lens: {target_audio_lens}")
        # print(f"target_audio_features: {target_audio_features}")
        # print(f"target_audio_features_lens: {target_audio_features_lens}")
        # print(f"text_tokens: {text_tokens}")
        # print(f"text_tokens_lens: {text_tokens_lens}")

        return {
            "utt_id": [cut.id for cut in cuts],
            "text": [cut.supervisions[0].text for cut in cuts],
            "audio": audio,
            "audio_lens": audio_lens,
            "audio_features": audio_features,
            "audio_features_lens": audio_features_lens,
            "target_audio": target_audio,
            "target_audio_lens": target_audio_lens,
            "target_audio_features": target_audio_features,
            "target_audio_features_lens": target_audio_features_lens,
            "text_tokens": text_tokens,
            "text_tokens_lens": text_tokens_lens,
        }

def validate_for_audio_to_audio(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        if cut.custom is None:
            raise ValueError(f"The 'custom' attribute of the cut object is None. It should be initialized as an empty dictionary. for cut: {cut}")
        if 'target_recording' not in cut.custom:
            raise ValueError(f"'target_recording' key is missing in the 'custom' attribute of the cut object. {cut}")
        assert (
            'target_recording' in cut.custom
        ), "Each cut must have a 'target_recording' in its custom field."