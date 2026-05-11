"""
JAX backend — XLA-compiled inference, optimal on TPUs and for research models.

Supports:
  - Flax/Haiku model state dicts (via orbax checkpoint)
  - Raw JAX functions saved with jax.export / cloudpickle
  - Automatic jit + vmap for batching
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..core.base_backend import BaseBackend, ModelInfo
from ..core.config import InferenceConfig


class JAXBackend(BaseBackend):
    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)
        self._apply_fn: Optional[Callable] = None
        self._params: Any = None
        self._jit_fn: Optional[Callable] = None

    def load(self) -> None:
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX backend requires: pip install jax jaxlib")

        model_path = Path(self.config.model.path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        suffix = model_path.suffix.lower()

        if suffix == ".pkl":
            self._load_cloudpickle(model_path)
        elif model_path.is_dir() or suffix in (".msgpack", ".orbax"):
            self._load_orbax(model_path)
        else:
            raise ValueError(
                f"JAX backend supports .pkl (cloudpickle) or orbax checkpoints, got: {suffix}"
            )

        # Configure device
        device_str = self.config.device.type
        if device_str == "cpu":
            jax.config.update("jax_platform_name", "cpu")
        elif device_str.startswith("cuda") or device_str == "gpu":
            jax.config.update("jax_platform_name", "gpu")

        self._jit_fn = jax.jit(self._apply_fn)
        self._loaded = True

    def _load_cloudpickle(self, path: Path) -> None:
        try:
            import cloudpickle as pickle
        except ImportError:
            import pickle  # type: ignore

        with open(path, "rb") as f:
            data = pickle.load(f)

        if callable(data):
            # Pure function: fn(inputs) -> outputs
            self._apply_fn = data
            self._params = None
        elif isinstance(data, dict) and "apply_fn" in data:
            # {"apply_fn": fn, "params": params}
            self._apply_fn = lambda x: data["apply_fn"](data["params"], x)
        else:
            raise ValueError("Unrecognized JAX checkpoint format")

    def _load_orbax(self, path: Path) -> None:
        try:
            import orbax.checkpoint as ocp
        except ImportError:
            raise ImportError("Orbax checkpoints require: pip install orbax-checkpoint")

        checkpointer = ocp.StandardCheckpointer()
        self._params = checkpointer.restore(str(path))
        # params only — apply_fn must be provided via metadata
        apply_fn_path = self.config.model.metadata.get("apply_fn_path")
        if apply_fn_path:
            import importlib
            module_path, fn_name = apply_fn_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            raw_fn = getattr(module, fn_name)
            self._apply_fn = lambda x: raw_fn(self._params, x)
        else:
            raise ValueError(
                "For orbax checkpoints, provide model.metadata.apply_fn_path "
                "(e.g. 'my_package.model.apply')"
            )

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        import jax.numpy as jnp

        if len(inputs) == 1:
            key = next(iter(inputs))
            jax_input = jnp.asarray(inputs[key])
            out = self._jit_fn(jax_input)
        else:
            jax_inputs = {k: jnp.asarray(v) for k, v in inputs.items()}
            out = self._jit_fn(jax_inputs)

        if hasattr(out, "shape"):
            return {self.config.model.output_names[0]: np.asarray(out)}
        elif isinstance(out, (list, tuple)):
            return {
                name: np.asarray(o)
                for name, o in zip(self.config.model.output_names, out)
            }
        elif isinstance(out, dict):
            return {k: np.asarray(v) for k, v in out.items()}
        else:
            raise TypeError(f"Unexpected JAX output type: {type(out)}")

    def warmup(self) -> None:
        dummy = self._make_dummy_inputs()
        for _ in range(self.config.optimization.warmup_iterations):
            self.infer(dummy)

    def get_info(self) -> ModelInfo:
        try:
            import jax
            jax_ver = jax.__version__
            devices = [str(d) for d in jax.devices()]
        except Exception:
            jax_ver = "unknown"
            devices = []

        return ModelInfo(
            backend="jax",
            model_path=self.config.model.path,
            input_names=self.config.model.input_names,
            output_names=self.config.model.output_names,
            input_shapes={k: tuple(v) for k, v in self.config.model.input_shapes.items()},
            output_shapes={k: tuple(v) for k, v in self.config.model.output_shapes.items()},
            device=self.config.device.type,
            fp16=self.config.optimization.fp16,
            int8=False,
            extra={"jax_version": jax_ver, "jax_devices": devices},
        )

    def unload(self) -> None:
        self._apply_fn = None
        self._params = None
        self._jit_fn = None
        self._loaded = False
