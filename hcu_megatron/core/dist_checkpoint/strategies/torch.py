# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from pathlib import Path

import torch
from torch.distributed.checkpoint import (
    BytesStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
from hyckpt_torch import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata

from megatron.core.dist_checkpointing.mapping import (
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
)


class TorchDistLoadShardedStrategy():
    """Basic load strategy for the PyT Distributed format."""

    def load_tensors_metadata(self, checkpoint_dir: Path, metadata: Metadata = None):
        """Uses tensors metadata stored in the metadata file."""
        if metadata is None:
            fs_reader = FileSystemReader(checkpoint_dir)
            metadata = fs_reader.read_metadata()

        mcore_data = getattr(metadata, 'mcore_data', {})
        sharded_metadata = {}
        for k, tp in metadata.state_dict_metadata.items():
            if not isinstance(tp, TensorStorageMetadata):
                continue  # load only tensors

            # Regular tensor
            sharded_metadata[k] = ShardedTensor.from_rank_offsets(
                k, torch.empty(tp.size, **tp.properties.__dict__, device='meta')
            ).without_data()
        return sharded_metadata

    def load_sharded_metadata(self, checkpoint_dir: Path) -> ShardedStateDict:
        """Uses tensors and objects metadata stored in the metadata file."""
        fs_reader = FileSystemReader(checkpoint_dir)
        metadata = fs_reader.read_metadata()

        sharded_metadata = {}
        for metadata_key, storage_metadata in metadata.state_dict_metadata.items():
            if not isinstance(storage_metadata, BytesStorageMetadata):
                continue
            sh_obj = ShardedObject.empty_from_unique_key(metadata_key)
            sharded_metadata[sh_obj.unique_key] = sh_obj

        sharded_metadata.update(self.load_tensors_metadata(checkpoint_dir, metadata))
        return sharded_metadata
