"""CodeBERT encoder plus a flat 9-way classification head (Milestone 7.3).

Concepts
--------
* Encoder: ``microsoft/codebert-base``, a 12-layer transformer that turns a token
  sequence into one 768-number vector per token. Fine-tuning means continuing to
  train these pretrained weights on our labelled data.
* [CLS]: CodeBERT puts a special start token at position 0 of every sequence. Its
  output vector is trained to summarize the whole sequence, so the head reads only
  that vector.
* Head: dropout (randomly zeroing some numbers during training, which reduces
  overfitting) followed by ONE linear layer that maps 768 numbers to 9 raw scores
  ("logits"), one per class. The head is flat, not hierarchical (Master Plan v3,
  section 1, guardrail 3).
* Logits are raw scores, not probabilities. The training loss (class-weighted
  cross-entropy) applies the softmax itself and lives OUTSIDE this module, so the
  class weights from M7.2 plug in without touching the model.

Design choices
--------------
* No pooler. Hugging Face's RobertaModel can add a pooling layer (a dense layer
  plus tanh over [CLS], about 590k parameters). Plan v3 specifies a single linear
  layer over [CLS], so the pooler is left out. This is a recommendation; the
  accuracy difference is expected to be negligible but has not been measured.
* The constructor takes an already-built encoder instead of downloading one. Tests
  inject a tiny random encoder, so they need no network and no GPU; the real model
  is loaded through ``from_pretrained``. This is the same dependency-injection
  pattern as ``sdp.data.tokenization``.
* This is a trainable ``torch.nn.Module``, not a ``Predictor``. The serving wrapper
  (``CodeBERTPredictor``) comes in M9.
* Gradient checkpointing (recomputing activations during the backward pass instead
  of storing them) trades about 30% more compute for much less memory. It is a
  switch because the local 4 GB GPU benchmark decides whether it is needed.
"""

from typing import Self

import torch
from torch import nn
from transformers import RobertaModel

from sdp.data.taxonomy import NUM_LEAVES


class CodeBERTClassifier(nn.Module):
    """Encoder, dropout, and one linear layer over the [CLS] vector."""

    def __init__(
        self,
        encoder: RobertaModel,
        num_labels: int = NUM_LEAVES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_labels < 2:
            raise ValueError(f"num_labels must be >= 2, got {num_labels}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.encoder = encoder
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)

        # BERT/RoBERTa convention: small random weights, zero bias.
        nn.init.normal_(self.classifier.weight, mean=0.0, std=encoder.config.initializer_range)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, num_labels).

        ``attention_mask`` is 1 for real tokens and 0 for padding, as built by
        ``sdp.data.dataset.collate_batch``.
        """
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_vector = hidden[:, 0, :]
        return self.classifier(self.dropout(cls_vector))

    def enable_gradient_checkpointing(self) -> None:
        """Trade extra compute for lower activation memory (active in train mode only)."""
        self.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Total parameter count, or only those that currently receive gradients."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_labels: int = NUM_LEAVES,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        local_files_only: bool = False,
    ) -> Self:
        """Load the pretrained encoder (no pooler) and attach a freshly initialized head.

        Returns the model in train mode so the encoder and the head agree. Loading
        prints a harmless notice that the checkpoint's pooler weights were not used.
        """
        encoder = RobertaModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            local_files_only=local_files_only,
        )
        model = cls(encoder, num_labels=num_labels, dropout=dropout)
        if gradient_checkpointing:
            model.enable_gradient_checkpointing()
        model.train()
        return model
