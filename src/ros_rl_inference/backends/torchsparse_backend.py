"""
TorchSparse backend for sparse 3D point cloud segmentation models.

Inherits from PyTorchSourceBackend — model class and constructor kwargs come
from config, so any torchsparse-based model loads without code changes.

What this backend adds over PyTorchSourceBackend:
  - Point cloud voxelization: raw (N,4) numpy → SparseTensor
  - Overrides infer() to build the batch dict expected by MinkUNet-style models
  - Overrides _move_to_device() to use torchsparse's to_() API

Config extras (under model.metadata):
  module:             str   dotted path to model class
                            e.g. "src.models.torchsparse_sep.MinkUNetWithAuxDecoder"
  project_root:       str   path prepended to sys.path
  constructor_kwargs: dict  passed verbatim to the model class __init__
  state_dict_key:     str   optional key to extract from checkpoint dict
  strict_load:        bool  default True
  voxel_size:         float default 0.1
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from ..core.config import InferenceConfig
from ..core.base_backend import ModelInfo
from .pytorch_backend import PyTorchSourceBackend


class TorchSparseBackend(PyTorchSourceBackend):
    """
    PyTorchSourceBackend specialization for torchsparse sparse-conv models.

    Adds point-cloud voxelization so callers pass raw (N,4) points and receive
    per-point logits — no external preprocessing required.
    """

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)
        self._voxel_size: float = float(
            config.model.metadata.get("voxel_size", 0.1)
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        super().load()
        # Refresh in case metadata was patched after __init__
        self._voxel_size = float(
            self.config.model.metadata.get("voxel_size", 0.1)
        )

    # ------------------------------------------------------------------
    # Hook override
    # ------------------------------------------------------------------

    def _move_to_device(
        self, model: torch.nn.Module, device: torch.device
    ) -> None:
        # torchsparse models expose to_() instead of the standard to()
        model.to_(str(device))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        inputs["points"]: (N, 4) numpy — x, y, z, intensity
        Returns {"logits": (N, num_classes) float32 numpy}.
        """
        batch = self._preprocess(inputs["points"])

        with torch.inference_mode():
            out = self._model(batch, get_main=True, get_aux=False)

        return {"logits": out["main_output"].cpu().numpy()}

    def predict_labels(self, points: np.ndarray) -> np.ndarray:
        """Convenience wrapper: returns per-point class index (N,)."""
        return np.argmax(self.infer({"points": points})["logits"], axis=1).astype(np.uint8)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, points: np.ndarray) -> Dict[str, Any]:
        """
        points: (N, 4) — x, y, z, intensity
        Returns the batch dict expected by torchsparse MinkUNet-style models.
        """
        from torchsparse.utils.quantize import sparse_quantize
        from torchsparse.utils.collate import sparse_collate
        from torchsparse import SparseTensor

        xyz = points[:, :3].astype(np.float32)
        intensity = points[:, 3:4].astype(np.float32)

        # Feature vector: [intensity, x, y, z]  (mirrors feat_dup=True in Goose3D)
        feats_np = np.concatenate([intensity, xyz], axis=1)  # (N, 4)

        pc_vox = np.round(xyz / self._voxel_size).astype(np.float32)
        pc_vox -= pc_vox.min(0, keepdims=True)

        coords, indices, inverse_map = sparse_quantize(
            pc_vox, return_index=True, return_inverse=True
        )

        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats_np[indices], dtype=torch.float32)
        inverse_map = torch.tensor(inverse_map, dtype=torch.long)

        # sparse_collate adds the batch-index column: (N,3) → (N,4)
        sparse_input = sparse_collate([SparseTensor(coords=coords, feats=feats)])
        sparse_input = sparse_input.to(self._device)
        inverse_map = inverse_map.to(self._device)

        return {
            "sparse_input": sparse_input,
            "sparse_input_invmap": inverse_map,
        }

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_info(self) -> ModelInfo:
        info = super().get_info()
        info.backend = "torchsparse"
        info.input_names = ["points"]
        info.output_names = ["logits"]
        info.input_shapes = {"points": (-1, 4)}
        num_classes = int(
            self.config.model.metadata.get("constructor_kwargs", {}).get("num_classes", -1)
        )
        info.output_shapes = {"logits": (-1, num_classes)}
        info.extra["voxel_size"] = self._voxel_size
        return info
