from transformers import TrainerCallback
import os
import torch
from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers.utils import logging
from huggingface_hub import upload_file

logger = logging.get_logger(__name__)

class ClearMLCallback(TrainerCallback):
    def __init__(self, task):
        self.task = task
        self.logger = task.get_logger()
        self._initialized = False

    def setup(self, args, state, model):
        if self._initialized:
            return
        self.task.connect(vars(args), name="TrainingArguments")
        if hasattr(model, "config"):
            self.task.connect(model.config, name="ModelConfig")
        self._initialized = True

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if state.is_world_process_zero:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.task.get_logger().report_scalar(
                        title=k, series=k, value=v, iteration=state.global_step
                    )
            logger.info(f"Step: {state.global_step}, Logs: {logs}")


class SaveCustomWeightsCallback(TrainerCallback):
    """
    Callback to save specific non-PEFT trainable weights alongside PEFT adapters.
    Assumes the Trainer handles saving PEFT adapters automatically.
    Also explicitly uploads the custom weights file to the Hub if push_to_hub is enabled.
    """
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        model = kwargs["model"]

        if not hasattr(model, "base_model") or not hasattr(model.base_model, "model"):
             logger.warning("Model structure not PeftModel. Skipping saving custom weights.")
             return

        custom_state_dict = {}
        layers_to_save = {
            "heat_embedding": model.base_model.model.heat_embedding,
            "visual_heat_blocks": model.base_model.model.visual.heat_blocks,
        }

        for name, layer in layers_to_save.items():
            for param_name, param in layer.named_parameters():
                if param.requires_grad:
                    full_param_name = f"{name}.{param_name}"
                    custom_state_dict[full_param_name] = param.cpu().clone()

        if not custom_state_dict:
            logger.info("No custom trainable weights found to save.")
            return

        custom_weights_filename = "custom_trained_weights.pt"
        save_path = os.path.join(checkpoint_folder, custom_weights_filename)
        try:
            os.makedirs(checkpoint_folder, exist_ok=True)
            torch.save(custom_state_dict, save_path)
            logger.info(f"Custom trainable weights saved locally to {save_path}")

            # --- Explicit Hub Upload --- #
            if args.push_to_hub and state.is_world_process_zero:
                repo_id = args.hub_model_id
                if not repo_id:
                    logger.warning("push_to_hub is True, but hub_model_id is not set. Cannot upload custom weights.")
                    return

                target_path_in_repo = custom_weights_filename # Upload to root
                commit_message = f"Upload custom weights for step {state.global_step}"
                try:
                    logger.info(f"Attempting to upload {save_path} to {repo_id}/{target_path_in_repo}")
                    upload_file(
                        path_or_fileobj=save_path,
                        path_in_repo=target_path_in_repo,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=commit_message,
                        # token=args.hub_token # Use configured token
                    )
                    logger.info(f"Successfully uploaded custom weights to Hub repo {repo_id}")
                except Exception as e:
                    logger.error(f"Failed to upload custom weights to Hub: {e}")
            # --- End Explicit Hub Upload --- #

        except Exception as e:
            logger.error(f"Failed to save custom trainable weights locally to {save_path}: {e}")
