"""
Self-supervised pretraining for point pattern encoders.

Stage 1 of the thesis pipeline: learn spatial structure from unlabeled data
using masked cell reconstruction before fine-tuning on survival.
"""

from .masked_pretrain import pretrain_encoder, load_pretrained_weights

__all__ = ["pretrain_encoder", "load_pretrained_weights"]
