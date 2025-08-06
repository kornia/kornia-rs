from collections import OrderedDict
from collections.abc import Callable
from typing import Any
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda"


import torch, random, numpy as np
from transformers import set_seed

from safetensors import safe_open


class ActivationIntrospector:
    def __init__(self):
        self.clear()

    def clear(self):
        self.activations = {}
        self._counter_token_pos = -1

    def set_introspector(self, model_layers: OrderedDict):
        self.model_layers = model_layers

        def hook_fn(name, initial_layer: bool = False):
            def hook(module, input, output):
                if initial_layer:
                    self._counter_token_pos += 1

                if isinstance(output, tuple):
                    output = output[0]
                elif isinstance(output, torch.Tensor):
                    pass
                else:
                    print("Hook unknown type!!!", name, type(output))

                self.activations[name+f"_i{self._counter_token_pos}"] = output.detach().cpu()

            return hook
        
        first_layer = True
        for name, model_layer in model_layers.items():
            model_layer.register_forward_hook(hook_fn(name, initial_layer=first_layer))
            if first_layer:
                first_layer = False

    def compare_rust_activations(
            self, token_indices: list[int], rust_safetensor: safe_open,
            report: Callable[[torch.Tensor, torch.Tensor], Any],
            subset: list[str] | None = None
    ):
        """
        expects the keys between Rust and Python are the same with the same format for each token position
        """

        layers = OrderedDict()

        for i in token_indices:
            for name in self.model_layers:
                if subset is None or (subset is not None and name in subset):
                    layers[name+f"_i{i}"] = report(rust_safetensor.get_tensor(name+f"_i{i}"), self.activations[name+f"_i{i}"].cpu())

        return layers


