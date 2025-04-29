import torch
import os
from transformers import AutoProcessor, AutoConfig, PreTrainedModel
from peft import PeftModel
from transformers.utils import logging
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError
from omegaconf import OmegaConf
import re

from src.qwen2_5.fa_model import Qwen2_5_VLForConditionalGenerationWithHeatmap


def _get_base_model_kwargs(
    config_path: str | None,
    default_kwargs: dict
) -> dict:
    kwargs = default_kwargs.copy()
    if not config_path or not os.path.exists(config_path):
        return kwargs
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    if hasattr(cfg, 'model.kwargs'):
        loaded_kwargs = OmegaConf.to_container(cfg.model.kwargs, resolve=True)
        kwargs.update(loaded_kwargs)
        kwargs.update(default_kwargs)
    return kwargs

def _load_base_model(base_model_path: str, config_path: str | None, **kwargs) -> PreTrainedModel:
    effective_kwargs = _get_base_model_kwargs(config_path, kwargs)
    model = Qwen2_5_VLForConditionalGenerationWithHeatmap.from_pretrained(
        base_model_path, **effective_kwargs
    )
    return model

def _load_peft_adapters(
    base_model: PreTrainedModel,
    checkpoint_identifier: str,
    subfolder: str | None,
    revision: str | None,
    **kwargs
) -> PeftModel:
    model = PeftModel.from_pretrained(
        base_model, checkpoint_identifier, subfolder=subfolder, revision=revision, **kwargs
    )
    return model


def _find_custom_weights_path(
    checkpoint_identifier: str,
    revision: str | None = None,
    filename: str = "custom_trained_weights.pt"
) -> str | None:
    # Case 1: checkpoint_identifier is a local directory
    if os.path.isdir(checkpoint_identifier):
        local_path = os.path.join(checkpoint_identifier, filename)
        if os.path.exists(local_path):
            return local_path
        else:
            # Don't attempt Hub download if it was explicitly a local path that failed
            return None

    # Case 2: checkpoint_identifier is a Hub repo ID
    try:
        downloaded_path = hf_hub_download(
            repo_id=checkpoint_identifier,
            filename=filename,
            revision=revision,
            cache_dir=None,
        )
        return downloaded_path
    except EntryNotFoundError:
        return None # File not found on Hub
    except Exception:
        return None # Other Hub download errors

def _apply_custom_weights(
    model: PeftModel,
    weights_path: str
):
    custom_state_dict = torch.load(weights_path, map_location='cpu')
    base_model = model.base_model.model

    # Group weights by target module
    weights_by_module = {}
    visual_block_pattern = re.compile(r"visual\.blocks\.(\d+)\.(attn_cross|norm3|norm4)\.(.*)")

    for key, value in custom_state_dict.items():
        if key.startswith("heat_embedding."):
            original_key = key[len("heat_embedding."):]
            weights_by_module.setdefault("heat_embedding", {})[original_key] = value
        else:
            match = visual_block_pattern.match(key)
            if match:
                block_idx, module_name, param_name = match.groups()
                block_idx = int(block_idx)
                module_key = f"visual.blocks.{block_idx}.{module_name}"
                weights_by_module.setdefault(module_key, {})[param_name] = value
    # Load weights into corresponding modules
    loaded_modules = []
    for module_key, state_dict_to_load in weights_by_module.items():
        target_module = None
        if module_key == "heat_embedding":
            if hasattr(base_model, "heat_embedding"):
                target_module = base_model.heat_embedding
                print(f"Loaded heat_embedding weights to model: {module_key}")
        elif module_key.startswith("visual.blocks."):
            parts = module_key.split('.') # visual.blocks.{idx}.{name}
            if len(parts) == 4:
                block_idx = int(parts[2])
                module_name = parts[3]
                if hasattr(base_model, "visual") and hasattr(base_model.visual, "blocks") and block_idx < len(base_model.visual.blocks):
                    block = base_model.visual.blocks[block_idx]
                    target_module = getattr(block, module_name, None)
                    print(f"Loaded visual block weights to model: {module_key}")
        if target_module is not None:
            target_module.load_state_dict(state_dict_to_load, strict=False)
            loaded_modules.append(module_key)


def load_model_with_adapters(
    base_model_path: str,
    checkpoint_identifier: str,
    subfolder: str | None = None,
    revision: str | None = None,
    load_custom_weights: bool = True,
    config_path: str | None = None,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = True,
    **peft_kwargs
) -> PeftModel:

    base_model_load_kwargs = {
        "torch_dtype": getattr(torch, torch_dtype) if torch_dtype != "auto" else torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    base_model = _load_base_model(base_model_path, config_path, **base_model_load_kwargs)

    adapter_load_kwargs = {"device_map": device_map, **peft_kwargs}
    model = _load_peft_adapters(base_model, checkpoint_identifier, subfolder, revision, **adapter_load_kwargs)
    base_model_unwrapped = model.base_model.model

    # Set requires_grad for heat_embedding
    if hasattr(base_model_unwrapped, "heat_embedding"):
        for param in base_model_unwrapped.heat_embedding.parameters():
            param.requires_grad = True

    # Set requires_grad for attn_cross, norm3, norm4 in visual blocks
    if hasattr(base_model_unwrapped, "visual") and hasattr(base_model_unwrapped.visual, "blocks"):
        for i, block in enumerate(base_model_unwrapped.visual.blocks):
            updated_in_block = False
            for module_name in ["attn_cross", "norm3", "norm4"]:
                module_obj = getattr(block, module_name, None)
                if module_obj:
                    for param in module_obj.parameters():
                        param.requires_grad = True
                    updated_in_block = True


    if load_custom_weights:
        custom_weights_path = _find_custom_weights_path(checkpoint_identifier, revision)
        if custom_weights_path:
            _apply_custom_weights(model, custom_weights_path)

    return model

# --- Processor Loading Function --- #
def load_processor(model_path_or_id: str, **kwargs) -> AutoProcessor:
    processor = AutoProcessor.from_pretrained(model_path_or_id, **kwargs)
    return processor


# Example usage (can be called from another script or notebook):
# if __name__ == '__main__':
#     # Example: Load from local checkpoint
#     local_model = load_model_with_adapters(
#         base_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
#         checkpoint_identifier="./heat_qwen2.5-checkpoints/checkpoint-100",
#         config_path="./config/qwen2.5_heat.yaml", # Optional
#         device_map="cuda",
#         torch_dtype="bfloat16"
#     )
#     processor = load_processor("Qwen/Qwen2.5-VL-3B-Instruct")
#     print("Loaded model from local checkpoint.")

#     # Example: Load from Hugging Face Hub checkpoint
#     hub_model = load_model_with_adapters(
#         base_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
#         checkpoint_identifier="Archistrax/Qwen2-5-VL-heat-checkpoints",
#         subfolder="checkpoint-100",
#         device_map="cuda",
#         torch_dtype="bfloat16"
#     )
#     processor = load_processor("Archistrax/Qwen2-5-VL-heat-checkpoints")
#     print("Loaded model from Hub checkpoint.") 