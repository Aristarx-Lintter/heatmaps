import torch
import os
from transformers import AutoProcessor, AutoConfig, PreTrainedModel
from peft import PeftModel
from transformers.utils import logging
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from omegaconf import OmegaConf

from src.qwen2_5.fa_model import Qwen2_5_VLForConditionalGenerationWithHeatmap
# from src.experiment import Experiment # Uncomment if you load config structure from Experiment

logger = logging.get_logger(__name__)
logging.set_verbosity_info()

def _get_base_model_kwargs(
    config_path: str | None,
    default_kwargs: dict
) -> dict:
    kwargs = default_kwargs.copy()
    if not config_path or not os.path.exists(config_path):
        return kwargs
    try:
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        if hasattr(cfg, 'model.kwargs'):
            loaded_kwargs = OmegaConf.to_container(cfg.model.kwargs, resolve=True)
            kwargs.update(loaded_kwargs)
            # Ensure explicitly passed args override config
            kwargs.update(default_kwargs)
            logger.info(f"Loaded and updated base model kwargs from {config_path}")
    except Exception as e:
        logger.warning(f"Could not load base model kwargs from {config_path}: {e}. Using defaults.")
    return kwargs

def _load_base_model(
    base_model_path: str,
    config_path: str | None,
    **kwargs
) -> PreTrainedModel:
    effective_kwargs = _get_base_model_kwargs(config_path, kwargs)
    logger.info(f"Loading base model: {base_model_path} with kwargs: {effective_kwargs}")
    try:
        model = Qwen2_5_VLForConditionalGenerationWithHeatmap.from_pretrained(
            base_model_path, **effective_kwargs
        )
        logger.info("Base model loaded.")
        return model
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        raise

def _load_peft_adapters(
    base_model: PreTrainedModel,
    checkpoint_identifier: str,
    subfolder: str | None,
    revision: str | None,
    **kwargs
) -> PeftModel:
    logger.info(f"Loading PEFT adapters from: {checkpoint_identifier}" + (f"/{subfolder}" if subfolder else ""))
    try:
        model = PeftModel.from_pretrained(
            base_model, checkpoint_identifier, subfolder=subfolder, revision=revision, **kwargs
        )
        logger.info("PEFT adapters loaded.")
        return model
    except Exception as e:
        logger.error(f"Failed to load PEFT adapters: {e}")
        raise

def _find_custom_weights_path(
    checkpoint_identifier: str,
    subfolder: str | None,
    revision: str | None,
    filename: str = "custom_trained_weights.pt"
) -> str | None:
    # Try local path first
    if os.path.isdir(checkpoint_identifier):
        local_dir = os.path.join(checkpoint_identifier, subfolder) if subfolder else checkpoint_identifier
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            logger.info(f"Found local custom weights: {local_path}")
            return local_path
        else:
             # If local dir exists but file doesn't, don't try Hub
             logger.warning(f"Custom weights not found locally at {local_path}")
             return None

    # Try Hub download if not found locally or if identifier wasn't a local dir
    try:
        downloaded_path = hf_hub_download(
            repo_id=checkpoint_identifier,
            filename=filename, # Search at the root
            revision=revision,
            cache_dir=None,
        )
        logger.info(f"Downloaded custom weights from Hub: {filename} (repo: {checkpoint_identifier}, revision: {revision})")
        return downloaded_path
    except EntryNotFoundError:
         logger.warning(f"Custom weights '{filename}' not found on Hub repo '{checkpoint_identifier}' (revision: {revision}).")
         return None
    except Exception as e:
        logger.error(f"Error downloading custom weights '{filename}' from Hub repo '{checkpoint_identifier}': {e}")
        return None

def _apply_custom_weights(
    model: PeftModel,
    weights_path: str
):
    try:
        logger.info(f"Loading custom weights state_dict from: {weights_path}")
        custom_state_dict = torch.load(weights_path, map_location='cpu')
        keys_heat = {k.split('.', 1)[1]: v for k, v in custom_state_dict.items() if k.startswith("heat_embedding.")}
        keys_visual = {k.split('.', 1)[1]: v for k, v in custom_state_dict.items() if k.startswith("visual_last_block.")}

        if keys_heat:
            missing, unexpected = model.base_model.model.heat_embedding.load_state_dict(keys_heat, strict=False)
            if missing or unexpected: logger.warning(f"Apply Heat Embedding - Missing: {missing}, Unexpected: {unexpected}")
            logger.info("Applied custom weights to heat_embedding.")

        if keys_visual:
            missing, unexpected = model.base_model.model.visual.blocks[-1].load_state_dict(keys_visual, strict=False)
            if missing or unexpected: logger.warning(f"Apply Visual Last Block - Missing: {missing}, Unexpected: {unexpected}")
            logger.info("Applied custom weights to visual.blocks[-1].")

    except Exception as e:
        logger.error(f"Failed to load or apply custom weights from {weights_path}: {e}")

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

    # --- Explicitly set custom layers to trainable AFTER loading PEFT adapters --- #
    # This is necessary because PeftModel.from_pretrained freezes the base model.
    logger.info("Setting requires_grad=True for custom trained layers (heat_embedding, visual.blocks[-1])...")
    for param in model.base_model.model.heat_embedding.parameters():
         param.requires_grad = True
    for param in model.base_model.model.visual.blocks[-1].parameters():
         param.requires_grad = True
    logger.info("Custom layers set to trainable.")
    # ----------------------------------------------------------------------------- #

    if load_custom_weights:
        custom_weights_path = _find_custom_weights_path(checkpoint_identifier, subfolder, revision)
        if custom_weights_path:
            _apply_custom_weights(model, custom_weights_path)
        else:
             logger.info("Proceeding without custom weights as they were not found.")

    logger.info("Model loading complete.")
    return model

# --- Processor Loading Function --- #
def load_processor(model_path_or_id: str, **kwargs) -> AutoProcessor:
    try:
        processor = AutoProcessor.from_pretrained(model_path_or_id, **kwargs)
        logger.info(f"Processor loaded from {model_path_or_id}")
        return processor
    except Exception as e:
        logger.error(f"Failed to load processor from {model_path_or_id}: {e}")
        raise

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