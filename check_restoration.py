import fire
import torch
from transformers.utils import logging
from dotenv import load_dotenv

from src.train_tools.restoration import load_model_with_adapters, load_processor

load_dotenv()

logger = logging.get_logger(__name__)
logging.set_verbosity_info()

def check_model_restoration(
    base_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    repo_id: str = "Archistrax/Qwen2-5-VL-heat-checkpoints-TEST",
    revision: str = "main", # Or specify a commit hash if needed
    # subfolder: str | None = None, # Usually None when loading from root of pushed checkpoint
    device_map: str = "cuda", # Or 'auto', 'cpu'
    torch_dtype: str = "bfloat16" # Or 'auto', 'float16', 'float32'
):
    """
    Tests loading the model and processor from a Hugging Face Hub repository
    where a checkpoint (adapters + custom weights) was pushed.
    """
    logger.info(f"--- Starting Restoration Check --- ")
    logger.info(f"Base Model: {base_model_path}")
    logger.info(f"Repo ID: {repo_id}")
    logger.info(f"Revision: {revision}")

    try:
        # Load the model using the restoration function
        # subfolder is typically None because Trainer pushes content of checkpoint-X to the root
        model = load_model_with_adapters(
            base_model_path=base_model_path,
            checkpoint_identifier=repo_id,
            subfolder=None, # Important: Load from repo root
            revision=revision,
            load_custom_weights=True,
            config_path=None, # Assuming base model kwargs don't need config
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        logger.info("Model loaded successfully using load_model_with_adapters.")

        # Check if custom weights were loaded (by checking requires_grad status)
        # Note: requires_grad might be True even if weights weren't loaded correctly,
        # but it's a basic check.
        # Access an internal parameter like linear1.weight instead of the module's weight
        if hasattr(model.base_model.model.heat_embedding, 'linear1') and model.base_model.model.heat_embedding.linear1.weight.requires_grad:
             logger.info("heat_embedding.linear1.weight appears to be trainable (as expected after loading).")
        elif hasattr(model.base_model.model.heat_embedding, 'linear2') and model.base_model.model.heat_embedding.linear2.weight.requires_grad:
             logger.info("heat_embedding.linear2.weight appears to be trainable (as expected after loading).") # Fallback check
        else:
             logger.warning("No trainable weights found directly checkable within heat_embedding.")

        # This check should be correct as mlp.gate_proj.weight exists
        if model.base_model.model.visual.blocks[-1].mlp.gate_proj.weight.requires_grad:
             logger.info("visual.blocks[-1].mlp.gate_proj.weight appears to be trainable (as expected after loading).")
        else:
             logger.warning("visual.blocks[-1].mlp.gate_proj.weight does NOT appear to be trainable.")

        # Try loading the processor from the same repo
        processor = load_processor(repo_id, revision=revision)
        logger.info("Processor loaded successfully.")

        # Optional: Perform a simple generation/inference check
        logger.info("Performing a simple inference check...")
        try:
            # Construct a minimal input
            # This needs adjustment based on your processor and model requirements
            text = "hello"
            inputs = processor(text=text, return_tensors="pt").to(device_map)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=5)
            logger.info(f"Inference output: {processor.decode(output[0], skip_special_tokens=True)}")
            logger.info("Simple inference check successful.")
        except Exception as e:
            logger.error(f"Simple inference check failed: {e}")

        logger.info("--- Restoration Check Complete --- ")

    except Exception as e:
        logger.error(f"Restoration check failed: {e}", exc_info=True)
        logger.info("--- Restoration Check Failed --- ")

if __name__ == "__main__":
    fire.Fire(check_model_restoration) 