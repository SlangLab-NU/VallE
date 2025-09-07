import random
from typing import Iterator, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from icefall.utils import make_pad_mask
from torchmetrics.classification import MulticlassAccuracy

from valle.data.input_strategies import PromptedFeatures
from valle.modules.embedding import SinePositionalEmbedding, TokenEmbedding
from valle.modules.transformer import (
    AdaptiveLayerNorm,
    LayerNorm,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from .macros import NUM_AUDIO_TOKENS, NUM_TEXT_TOKENS

class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)

class ValleCore(nn.Module):
    """
    Contains the core elements needed for Vall-E models
     - __init__
     - pad_y_eos / _prepare_prompts helper functions
     - stage_(named_)parameters functions
    no forward(), inference, or continual methods
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        # HANDLING SEMANTIC EMBEDDINGS
        use_model_embeddings: str,
        embed_dim: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        decoder_cls=nn.TransformerEncoder,
        decoder_layer_cls=nn.TransformerEncoderLayer,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        prepend_bos: bool = False,
        num_quantizers: int = 8,
    ):
        super().__init__()
    
        # -------- Hyperparameters ------------

        self.rng = random.Random(0)
        self.num_heads = nhead
        self.prefix_mode = prefix_mode
        self.num_quantizers = num_quantizers
        self.ar_audio_prepend_bos = prepend_bos

        # ------ Embeddings --------

        nar_d_model = int(d_model * nar_scale_factor)

        if use_model_embeddings == "whisper" or use_model_embeddings == "wavlm":
            # Replacing the TokenEmbedding with Linear projection layer
            self.ar_text_embedding = nn.Sequential(
                nn.Linear(embed_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1)
            )
            self.nar_text_embedding = nn.Sequential(
                nn.Linear(embed_dim, nar_d_model),
                nn.LayerNorm(nar_d_model),
                nn.Dropout(0.1)
            )
        else:
            self.ar_text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)  # W_x
            self.nar_text_embedding = TokenEmbedding(nar_d_model, NUM_TEXT_TOKENS)

        # ----- Build AR Decoder ---------

        # ID NUM_AUDIO_TOKENS     -> PAD
        # ID NUM_AUDIO_TOKENS + 1 -> BOS
        self.ar_audio_embedding = TokenEmbedding(
            d_model, NUM_AUDIO_TOKENS + 1 + int(prepend_bos)
        )
        
        self.ar_text_prenet, self.ar_audio_prenet = build_prenets(
                d_model, add_prenet
        )

        self.ar_text_position  = SinePositionalEmbedding(d_model,  dropout=0.1, alpha=True)
        self.ar_audio_position = SinePositionalEmbedding(d_model,  dropout=0.1, alpha=True)

        self.ar_decoder = decoder_cls(
            decoder_layer_cls(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )
        
        self.ar_predict_layer = nn.Linear(d_model, NUM_AUDIO_TOKENS + 1, bias=False)

        self.ar_accuracy_metric = MulticlassAccuracy(
            NUM_AUDIO_TOKENS + 1,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=NUM_AUDIO_TOKENS,
        )

        # -------- NAR Decoder -----------

        assert num_quantizers >= 1
        if num_quantizers > 1:
            self.nar_audio_embeddings = nn.ModuleList(
                [TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS + 1)]
                + [
                    TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS)
                    for i in range(num_quantizers - 1)
                ]
            )  # W_a

            self.nar_text_prenet, self.nar_audio_prenet = build_prenets(
                    nar_d_model, add_prenet
            )
      
            self.nar_text_position = SinePositionalEmbedding(nar_d_model, dropout=0.0, alpha=False)
            self.nar_audio_position= SinePositionalEmbedding(nar_d_model, dropout=0.1, alpha=False)

            self.nar_decoder = decoder_cls(
                decoder_layer_cls(
                    nar_d_model,
                    int(nhead * nar_scale_factor),
                    dim_feedforward=nar_d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=norm_first,
                    adaptive_layer_norm=True,
                ),
                num_layers=int(num_layers * nar_scale_factor),
                norm=AdaptiveLayerNorm(
                    nar_d_model, norm=nn.LayerNorm(nar_d_model)
                )
                if norm_first
                else None,
            )
            self.nar_predict_layers = nn.ModuleList(
                [
                    nn.Linear(nar_d_model, NUM_AUDIO_TOKENS, bias=False)
                    for i in range(num_quantizers - 1)
                ]
            )
            self.nar_stage_embeddings = nn.ModuleList(
                [
                    TokenEmbedding(nar_d_model, 1)
                    for i in range(num_quantizers - 1)
                ]
            )

            if share_embedding:
                # We share the parameters of the output projection layer with the parameters of the acoustic embedding Wa
                # NOTE(Feiteng): In the experiment, this undermines accuracy
                # self.ar_predict_layer.weight = self.ar_audio_embedding.weight

                # We also share the parameters of the acoustic embedding layer and the output prediction layer,
                # which means the weights of the j-th prediction layer are the same as the (j + 1)-th acoustic embedding layer.
                for j in range(0, num_quantizers - 2):
                    self.nar_predict_layers[
                        j
                    ].weight = self.nar_audio_embeddings[j + 2].weight

            self.nar_accuracy_metric = MulticlassAccuracy(
                NUM_AUDIO_TOKENS + 1,
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=NUM_AUDIO_TOKENS,
            )

    # --------- Helper Methods ------------

    def stage_parameters(self, stage: int = 1) -> Iterator[nn.Parameter]:
        assert stage > 0
        if stage == 1:
            for name, param in self.named_parameters():
                if name.startswith("ar_"):
                    print(f" AR parameter: {name}")
                    yield param

        if stage == 2:
            for name, param in self.named_parameters():
                if name.startswith("nar_"):
                    print(f"NAR parameter: {name}")
                    yield param

    def stage_named_parameters(
        self, stage: int = 1
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        assert stage > 0
        if stage == 1:
            for pair in self.named_parameters():
                if pair[0].startswith("ar_"):
                    yield pair

        if stage == 2:
            for pair in self.named_parameters():
                if pair[0].startswith("nar_"):
                    yield pair

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(
            y_mask_int, (0, 1), value=1
        )
        # inputs, targets
        if self.ar_audio_prepend_bos:
            return (
                F.pad(targets[:, :-1], (1, 0), value=NUM_AUDIO_TOKENS + 1),
                targets,
            )

        return targets[:, :-1], targets[:, 1:]

    def _prepare_prompts(self, y, y_lens, codes, nar_stage, y_prompts_codes, max_atypical_len=None):
        # 5.1 For the NAR acoustic prompt tokens, we select a random segment waveform of 3 seconds
        # from the same utterance.
        # We implement this differently.
        if self.prefix_mode == 0:
            # no prefix
            prefix_len = 0
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, nar_stage):
                # Formula (4) (5)
                y_emb = y_emb + self.nar_audio_embeddings[j](codes[..., j])
        elif self.prefix_mode == 1:
            # prefix at begining
            int_low = (0.25 * y_lens.min()).type(torch.int64).item()
            prefix_len = torch.randint(int_low, int_low * 2, size=()).item()
            prefix_len = min(prefix_len, 225)  # 24000/320 * 3s = 225 frames

            y_prompts = self.nar_audio_embeddings[0](y[:, :prefix_len])
            y_emb = self.nar_audio_embeddings[0](y[:, prefix_len:])
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](
                    codes[:, :prefix_len, j]
                )
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](
                        codes[:, prefix_len:, j]
                    )
            y_emb = torch.concat([y_prompts, y_emb], axis=1)

        # TODO Change mode to 5. Remove the prefix_len and make it max_atypical_len
        elif self.prefix_mode == 5:
            # prefix at begining
            prefix_len = max_atypical_len

            y_prompts = self.nar_audio_embeddings[0](y[:, :prefix_len])
            y_emb = self.nar_audio_embeddings[0](y[:, prefix_len:])
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](
                    codes[:, :prefix_len, j]
                )
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](
                        codes[:, prefix_len:, j]
                    )
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        elif self.prefix_mode in [2, 4]:
            if self.prefix_mode == 2:
                # random prefix
                prefix_len = min(225, int(0.25 * y_lens.min().item()))

                y_prompts_codes = []
                for b in range(codes.shape[0]):
                    start = self.rng.randint(0, y_lens[b].item() - prefix_len)
                    y_prompts_codes.append(
                        torch.clone(codes[b, start : start + prefix_len])
                    )
                    codes[
                        b, start : start + prefix_len, nar_stage
                    ] = NUM_AUDIO_TOKENS
                y_prompts_codes = torch.stack(y_prompts_codes, dim=0)
            else:
                prefix_len = y_prompts_codes.shape[1]

            y_prompts = self.nar_audio_embeddings[0](y_prompts_codes[..., 0])
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](
                    y_prompts_codes[..., j]
                )
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](codes[..., j])
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        else:
            raise ValueError

        return y_emb, prefix_len


def build_prenets(d_model: int, add: bool):
    if not add:
        return nn.Identity(), nn.Identity()
    return (
        nn.Sequential(
            Transpose(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
            nn.BatchNorm1d(d_model),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
            nn.BatchNorm1d(d_model),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
            nn.BatchNorm1d(d_model),
            nn.ReLU(), nn.Dropout(0.5),
            Transpose(),
            nn.Linear(d_model, d_model),
        ),
        nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 256),
            nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, d_model),
        ),
    )