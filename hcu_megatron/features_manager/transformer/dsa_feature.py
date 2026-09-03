# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser

from hcu_megatron.features_manager.feature import AbstractFeature


class DSAFeature(AbstractFeature):

    def __init__(self):
        super().__init__('dsa', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        pass

    def register_patches(self, patch_manager, args):
        from hcu_megatron.core.transformer.experimental_attention_variant.dsa import (
            DSAIndexer,
            dsa_attention_forward_wrapper,
        )

        patch_manager.register_cls_funcs(
            'megatron.core.transformer.experimental_attention_variant.dsa.DSAIndexer.forward_before_topk',
            [DSAIndexer.forward_before_topk,
             DSAIndexer.forward_with_scores,]
        )
        patch_manager.register_patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.DSAttention.forward',
            dsa_attention_forward_wrapper,
            apply_wrapper=True,
        )
