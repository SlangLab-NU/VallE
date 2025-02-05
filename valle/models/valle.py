# Copyright    2023                             (authors: Feiteng Li)
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

import random
from typing import Dict, Iterator, List, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from icefall.utils import make_pad_mask
from torchmetrics.classification import MulticlassAccuracy
from valle.data.input_strategies import PromptedFeatures
from valle.modules.embedding import SinePositionalEmbedding, TokenEmbedding, ContinuousEmbedding
from valle.modules.transformer import (
    AdaptiveLayerNorm,
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer
)

from .macros import NUM_AUDIO_TOKENS, NUM_TEXT_TOKENS
from .valle_helper import topk_sampling
from .visualizer import visualize


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)


# NOTE: There are two ways to implement the model
#       1) [VALL-F] standard TransformerDecoder, use x as memory
#       2) [VALL-E] modified TransformerDecoder like GPT-x(e.g. causal TransformerEncoder),
#          use x as the prefix of decoder inputs

class VALLE(nn.Module):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """
    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_layers: int,
            norm_first: bool = True,
            decoder_cls: type = TransformerEncoder,
            decoder_layer_cls: type = TransformerEncoderLayer,
            prefix_mode: int = 0,
            share_embedding: bool = True,
            nar_scale_factor: float = 1.0,
            prepend_bos: bool = False,
            num_quantizers: int = 8,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super().__init__()
        nar_d_model = int(d_model * nar_scale_factor)

        self.ar_text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)  # W_x
        self.nar_text_embedding = TokenEmbedding(nar_d_model, NUM_TEXT_TOKENS)

        # ID NUM_AUDIO_TOKENS     -> PAD
        # ID NUM_AUDIO_TOKENS + 1 -> BOS
        self.ar_audio_prepend_bos = prepend_bos

        self.ar_audio_embedding = TokenEmbedding(
            d_model, NUM_AUDIO_TOKENS + 1 + int(prepend_bos)
        )

        self.ar_text_prenet = nn.Identity()
        self.ar_audio_prenet = nn.Identity()

        self.ar_text_position, self.ar_audio_position = self._create_positional_embeddings(d_model)

        ar_norm = LayerNorm(d_model) if norm_first else None
        self.ar_decoder = self._create_decoder(
            decoder_cls, decoder_layer_cls, d_model, nhead, num_layers, ar_norm
        )
        self.ar_predict_layer = nn.Linear(
            d_model, NUM_AUDIO_TOKENS + 1, bias=False
        )

        self.ar_accuracy_metric = MulticlassAccuracy(
            NUM_AUDIO_TOKENS + 1,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=NUM_AUDIO_TOKENS,
        )

        self.rng = random.Random(0)
        self.num_heads = nhead
        self.prefix_mode = prefix_mode
        self.num_quantizers = num_quantizers

        assert num_quantizers >= 1
        if num_quantizers > 1:
            self._initialize_nar_components(nar_d_model, nhead, num_layers, norm_first, decoder_cls, decoder_layer_cls,
                                            share_embedding, nar_scale_factor)

    def _create_positional_embeddings(self, d_model, dropout_text=0.1, dropout_audio=0.1):
        """ Creates positional embeddings for AR text and audio. """
        return (
            SinePositionalEmbedding(d_model, dropout=dropout_text, scale=False, alpha=True),
            SinePositionalEmbedding(d_model, dropout=dropout_audio, scale=False, alpha=True),
        )

    def _create_decoder(self, decoder_cls, decoder_layer_cls, d_model, nhead, num_layers, norm, norm_first):
        """ Creates a transformer decoder. """
        return decoder_cls(
            decoder_layer_cls(d_model, nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True,
                              norm_first=norm_first),
            num_layers=num_layers,
            norm=norm,
        )

    def _initialize_nar_components(self, nar_d_model, nhead, num_layers, norm_first, decoder_cls, decoder_layer_cls,
                                   share_embedding, nar_scale_factor):
        """ Initializes Non-Auto-Regressive (NAR) components. """
        self.nar_audio_embeddings = nn.ModuleList(
            [TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS + 1)]
            + [TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS) for _ in range(self.num_quantizers - 1)]
        )

        self.nar_text_prenet, self.nar_audio_prenet = nn.Identity(), nn.Identity()

        self.nar_text_position, self.nar_audio_position = self._create_positional_embeddings(nar_d_model, dropout_text=0.0)

        nar_norm = AdaptiveLayerNorm(nar_d_model, norm=nn.LayerNorm(nar_d_model)) if norm_first else None
        self.nar_decoder = self._create_decoder(decoder_cls, decoder_layer_cls, nar_d_model,
                                                int(nhead * nar_scale_factor),
                                                num_layers=int(num_layers * nar_scale_factor),
                                                norm=nar_norm, norm_first=norm_first)

        self.nar_predict_layers = nn.ModuleList(
            [nn.Linear(nar_d_model, NUM_AUDIO_TOKENS, bias=False) for _ in range(self.num_quantizers - 1)])

        self.nar_stage_embeddings = nn.ModuleList(
            [TokenEmbedding(nar_d_model, 1) for _ in range(self.num_quantizers - 1)])

        if share_embedding:
            for j in range(self.num_quantizers - 2):
                self.nar_predict_layers[j].weight = self.nar_audio_embeddings[j + 2].weight

        self.nar_accuracy_metric = MulticlassAccuracy(
            NUM_AUDIO_TOKENS + 1, top_k=10, average="micro", multidim_average="global", ignore_index=NUM_AUDIO_TOKENS
        )

    def _inference_step(self, x, x_lens, y, top_k, temperature, nar_decoder=True):
        """Handles inference logic inside forward."""
        text_len = x_lens.max()
        prefix_len =y.shape[1]
        y = y[..., 0]
        prompts = y

        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)

        x_attn_mask = torch.zeros((text_len, text_len), dtype=torch.bool)

        while True:
            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(x_attn_mask, (0, y_len), value=True)
            y_attn_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1), (text_len, 0),
                                value=False)
            xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).to(y.device)
            xy_dec = self._ar_decoder_forward(x, x_lens,y, xy_attn_mask)
            y_emb = self.nar_audio_embeddings[0](
                y[:, int(self.ar_audio_prepend_bos):]
            )
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(logits, top_k=top_k, top_p=1.0, temperature=temperature)

            #some of the original checks are missing
            if samples[0, 0] == NUM_AUDIO_TOKENS or y.shape[1] > x_lens.max() * 16:
                break

            y = torch.concat([y, samples], dim=1)

        #TODO: Clarify what this line does
        codes = [y[:, prefix_len + int(self.ar_audio_prepend_bos):]]
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)

        # Non-AR Decoders
        y_emb = self.nar_audio_embeddings[0](
                y[:, int(self.ar_audio_prepend_bos):]
            )
        x = self.nar_text_embedding(x)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        if self.prefix_mode is not 0:
            for j in range(1, self.num_quantizers):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](
                    prompts[..., j]
                )

        for i, (predict_layer, embedding_layer) in enumerate(
            zip(
                self.nar_predict_layers,
                self.nar_audio_embeddings[1:],
            )
        ):
            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = self.nar_audio_position(y_pos)
            xy_pos = torch.concat([x, y_pos], dim=1)

            xy_dec, _ = self.nar_decoder(
                (xy_pos, self.nar_stage_embeddings[i].weight)
            )
            logits = predict_layer(xy_dec[:, text_len + prefix_len :])

            samples = torch.argmax(logits, dim=-1)
            codes.append(samples)

            if i < self.num_quantizers - 2:
                if self.prefix_mode !=0:
                    y_emb[:, :prefix_len] += embedding_layer(
                                                prompts[..., i + 1]
                                                )
                y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)

    def _compute_attention_masks(self, x_len, y_len, x_device):
        x_attn_mask = F.pad(
            torch.zeros((x_len, x_len), dtype=torch.bool, device=x_device),
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(
                torch.ones(y_len, y_len, dtype=torch.bool, device=x_device),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )
        return torch.concat([x_attn_mask, y_attn_mask], dim=0)

    def _ar_decoder_forward(self, x, x_lens, y, xy_attn_mask):
        y_emb = self.ar_audio_embedding(y)
        y_emb = self.ar_audio_prenet(y_emb)
        y_pos = self.ar_audio_position(y_emb)

        xy_pos = torch.concat([x, y_pos], dim=1)

        xy_dec, _ = self.ar_decoder(
            (xy_pos, None),
            mask=xy_attn_mask,
            # src_key_padding_mask=xy_padding_mask,
            # is_causal=True,
        )
        return xy_dec, y_emb

    def _nar_decoder_forward(
            self,
            x,
            x_lens,
            y,
            y_lens,
            codes,
            y_prompts_codes,
            xy_padding_mask,
            x_mask,
            y_mask
            ):
        text = x
        num_nar_layers = self.num_quantizers - 1
        nar_stage = self.rng.choices(
            [_k for _k in range(1, self.num_quantizers)],
            weights=[1.0 / num_nar_layers] * num_nar_layers,
            k=1,
        )[0]

        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        y_emb, prefix_len = self._prepare_prompts(
            y, y_lens, codes, nar_stage, y_prompts_codes
        )

        y_pos = self.nar_audio_prenet(y_emb)
        y_pos = self.nar_audio_position(y_pos)
        xy_pos = torch.concat([x, y_pos], dim=1)

        y_len = y_lens.max()
        targets = codes[..., nar_stage] + NUM_AUDIO_TOKENS * y_mask.type(torch.int64)

        if self.prefix_mode in [2, 4]:
            xy_padding_mask = torch.concat(
                [
                    x_mask,
                    F.pad(y_mask, (y_emb.shape[1] - y_len, 0), value=False),
                ],
                dim=1,
            )
        elif self.prefix_mode == 1:
            targets = targets[:, prefix_len:]

        xy_dec, _ = self.nar_decoder(
            (xy_pos, self.nar_stage_embeddings[nar_stage - 1].weight),
            src_key_padding_mask=xy_padding_mask,
            # is_causal=False,
        )

        xy_dec = xy_dec[:, x_lens.max() + prefix_len:]
        if self.prefix_mode == 4:
            prefix_len = 0  # reset for Top10Accuracy metric
        logits = self.nar_predict_layers[nar_stage - 1](xy_dec).permute(
            0, 2, 1
        )

        return logits, prefix_len, targets


    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: Union[torch.Tensor, PromptedFeatures],
        y_lens: Union[torch.Tensor, PromptedFeatures, None],
        reduction: str = "sum",
        train_stage: int = 0,
        top_k: int = -100,
        temperature: float = 1.0,
        **kwargs,
    ) -> Tuple[Tuple[Any, Union[torch.Tensor, Any]], Union[torch.Tensor, float], dict[str, Union[torch.Tensor, Any]]]:
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, T, 8).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          train_stage:
            0: AR & NAR modules, 1: AR modules, 2: NAR modules
        Returns:
          Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert torch.all(x_lens > 0)
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        assert y_lens.ndim == 1, y_lens.shape
        y_prompts_codes = None
        if isinstance(y, PromptedFeatures):
            y_prompts_codes, y = y.data
            prompts_len, y_lens = y_lens.data
            assert prompts_len.min() == prompts_len.max()
            assert self.prefix_mode == 4
            y_prompts_codes = y_prompts_codes.type(torch.int64)

        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)
        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))
        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        y, targets = self.pad_y_eos(
            codes[..., 0], y_mask_int, eos_id=NUM_AUDIO_TOKENS
        )

        x_len = x_lens.max()
        y_len = y_lens.max()
        metrics = {}
        total_loss = 0.0

        if self.ar_audio_prepend_bos:
            ar_xy_padding_mask = torch.concat(
                [x_mask, F.pad(y_mask, (1, 0), value=False)], dim=1
            )
            print(f"BOS prepended: {ar_xy_padding_mask}")
        else:
            ar_xy_padding_mask = xy_padding_mask
            print(f"BOS NOT prepended: {ar_xy_padding_mask}")
        # AR Decoder
        if train_stage in [0, 1]:
            xy_attn_mask = self._compute_attention_masks(x_len, y_len, x.device)
            # merge key padding and attention masks
            bsz, src_len = x.shape[0], x_len + y_len
            _xy_padding_mask = (
                ar_xy_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(bsz * self.num_heads, 1, src_len)
            )
            xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)

            new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
            new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
            xy_attn_mask = new_attn_mask

            xy_dec = self._ar_decoder_forward(x, x_lens, y, xy_attn_mask)
            logits = self.ar_predict_layer(xy_dec[:, x_lens.max():]).permute(0, 2, 1)
            # loss
            total_loss = F.cross_entropy(logits, targets, reduction=reduction)

            metrics["ArTop10Accuracy"] = self.ar_accuracy_metric(
                logits.detach(), targets
            ).item() * y_lens.sum().type(torch.float32)
        print(f"Codes from AR forward: {codes}")
        if self.num_quantizers == 1:
            return (x, codes), total_loss, metrics

        # Non-AR Decoders
        if self.ar_audio_prepend_bos:
            y = y[:, 1:]
        if train_stage in [0, 2]:
            logits, prefix_len, targets = self._nar_decoder_forward(
                x,
                x_lens,
                y,
                y_lens,
                codes,
                y_prompts_codes,
                xy_padding_mask,
                x_mask,
                y_mask)

            # loss
            total_length = y_lens.sum().type(torch.float32)
            total_loss += (
                F.cross_entropy(
                    logits,
                    targets,
                    ignore_index=NUM_AUDIO_TOKENS,
                    reduction=reduction,
                )
                * (total_length / (total_length - prefix_len * x.shape[0]))
            )
            metrics["NarTop10Accuracy"] = (
                self.nar_accuracy_metric(
                    F.pad(
                        logits.detach(),
                        (0, 0, 0, 1, 0, 0),
                        value=logits.min().cpu().item(),
                    ),
                    targets,
                ).item()
                * total_length
            )

        if train_stage == 0:
            total_loss = total_loss / 2.0

        return (x, codes), total_loss, metrics

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: torch.Tensor = None,
        top_k: int = -100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
          Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert torch.all(x_lens > 0)
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        return self._inference_step(x, x_lens, y, top_k, temperature)

    def continual(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, S).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, 8).
        Returns:
          Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)
        assert self.num_quantizers == 8

        # NOTE: x has been padded in TextTokenCollater
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()

        prefix_len = min(int(y.shape[1] * 0.5), 3 * 75)

        # AR Decoder
        prompts = y[:, :prefix_len]

        codes = [y[:, prefix_len:, 0]]
        # Non-AR Decoders
        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        y_emb = self.nar_audio_embeddings[0](y[..., 0])

        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_position(y_emb)
                y_pos = self.nar_audio_prenet(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < 6:
                    y_emb[:, :prefix_len] += embedding_layer(
                        prompts[..., i + 1]
                    )
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, 8):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](
                    prompts[..., j]
                )

            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < 6:
                    y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == 8
        return torch.stack(codes, dim=-1)

