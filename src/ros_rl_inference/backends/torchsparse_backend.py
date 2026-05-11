"""
TorchSparse backend for sparse 3D point cloud segmentation models.

Handles the full pipeline:
  raw (N,4) numpy → voxelization → SparseTensor → model → per-point logits

Designed for MinkUNet-style models (off-road-dg-lss MinkUNetWithAuxDecoder).
Requires torchsparse to be installed (use the blopausore/torchsparse Docker image).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..core.base_backend import BaseBackend, ModelInfo
from ..core.config import InferenceConfig


class TorchSparseBackend(BaseBackend):
    """
    Backend for torchsparse sparse-conv models.

    Config extras (under model.metadata):
      num_classes:                  int   (required)
      num_auxiliary_classes:        int   (default 10)
      voxel_size:                   float (default 0.1)
      cr:                           float (default 1.0)
      use_spatial_context:          bool  (default False)
      spatial_context_out_channels: int   (default 16)
      cylindrical_coordinates:      bool  (default False)
      project_root:                 str   path to add to sys.path so model
                                          source code can be imported
    """

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)
        self._model = None
        self._device: Optional[torch.device] = None
        self._voxel_size: float = 0.1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        meta = self.config.model.metadata

        # Add project root to sys.path so the model sources are importable
        project_root = meta.get("project_root")
        if project_root and project_root not in sys.path:
            sys.path.insert(0, project_root)

        try:
            from src.models.torchsparse_sep import MinkUNetWithAuxDecoder
        except ImportError as e:
            raise ImportError(
                "Could not import MinkUNetWithAuxDecoder. "
                "Set model.metadata.project_root to the off-road-dg-lss directory "
                f"or run inside the torchsparse Docker container.\n{e}"
            )

        num_classes = int(meta.get("num_classes", 64))
        num_aux = int(meta.get("num_auxiliary_classes", 10))
        self._voxel_size = float(meta.get("voxel_size", 0.1))
        cr = float(meta.get("cr", 1.0))
        use_spatial_context = bool(meta.get("use_spatial_context", False))
        spatial_out_ch = int(meta.get("spatial_context_out_channels", 16))
        cylindrical = bool(meta.get("cylindrical_coordinates", False))

        self._device = torch.device(self.config.device.type)

        model = MinkUNetWithAuxDecoder(
            in_features=4,
            num_classes=num_classes,
            num_auxiliary_classes=num_aux,
            voxel_size=self._voxel_size,
            cylindrical_coordinates=cylindrical,
            cr=cr,
            use_spatial_context=use_spatial_context,
            spatial_context_out_channels=spatial_out_ch,
        )

        model_path = Path(self.config.model.path)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        state_dict = torch.load(str(model_path), map_location=self._device)
        model.load_state_dict(state_dict)
        model.to_(str(self._device))
        model.eval()

        self._model = model
        self._loaded = True

    def warmup(self) -> None:
        pass  # torchsparse JIT happens on first real call

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        inputs must contain key "points": numpy array of shape (N, 4)
        with columns [x, y, z, intensity].

        Returns dict with key "logits": (N, num_classes) float32 numpy array.
        """
        points = inputs["points"]  # (N, 4): x, y, z, intensity
        batch = self._preprocess(points)

        with torch.inference_mode():
            out = self._model(batch, get_main=True, get_aux=False)

        logits = out["main_output"].cpu().numpy()  # (N, num_classes)
        return {"logits": logits}

    def predict_labels(self, points: np.ndarray) -> np.ndarray:
        """Convenience wrapper: returns per-point class index (N,)."""
        out = self.infer({"points": points})
        return np.argmax(out["logits"], axis=1).astype(np.uint8)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, points: np.ndarray) -> Dict[str, Any]:
        """
        points: (N, 4) — x, y, z, intensity
        Returns the batch dict expected by MinkUNetWithAuxDecoder.
        """
        from torchsparse.utils.quantize import sparse_quantize
        from torchsparse.utils.collate import sparse_collate
        from torchsparse import SparseTensor

        xyz = points[:, :3].astype(np.float32)
        intensity = points[:, 3:4].astype(np.float32)

        # Feature vector: [intensity, x, y, z]  (mirrors feat_dup=True in Goose3D)
        feats_np = np.concatenate([intensity, xyz], axis=1)  # (N, 4)

        # Voxelize
        pc_vox = np.round(xyz / self._voxel_size).astype(np.float32)
        pc_vox -= pc_vox.min(0, keepdims=True)

        coords, indices, inverse_map = sparse_quantize(
            pc_vox, return_index=True, return_inverse=True
        )

        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats_np[indices], dtype=torch.float32)
        inverse_map = torch.tensor(inverse_map, dtype=torch.long)

        # sparse_collate adds the batch-index column (N,3) → (N,4)
        sparse_input = sparse_collate([SparseTensor(coords=coords, feats=feats)])
        sparse_input = sparse_input.to(self._device)
        inverse_map = inverse_map.to(self._device)

        return {
            "sparse_input": sparse_input,
            "sparse_input_invmap": inverse_map,
        }

    # ------------------------------------------------------------------
    # Info / cleanup
    # ------------------------------------------------------------------

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            backend="torchsparse",
            model_path=self.config.model.path,
            input_names=["points"],
            output_names=["logits"],
            input_shapes={"points": (-1, 4)},
            output_shapes={"logits": (-1, int(self.config.model.metadata.get("num_classes", 64)))},
            device=self.config.device.type,
            fp16=False,
            int8=False,
            extra={
                "voxel_size": self._voxel_size,
                "use_spatial_context": self.config.model.metadata.get("use_spatial_context"),
                "num_classes": self.config.model.metadata.get("num_classes"),
            },
        )

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()
