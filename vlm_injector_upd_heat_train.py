import fire
import os
import math

from omegaconf import OmegaConf
import torch
from transformers import AutoProcessor, AutoConfig, PreTrainedModel, ProcessorMixin, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers.utils import logging

from src.common.dataset import DataCollator
from src.experiment import Experiment
from src.qwen2_5.fa_model import Qwen2_5_VLForConditionalGenerationWithHeatmap
from src.train_tools.callbacks import ClearMLCallback, SaveCustomWeightsCallback
from src.train_tools.initiator import init_transformer_block_weights
from src.train_tools.restoration import _apply_custom_weights, _find_custom_weights_path, load_processor

logger = logging.get_logger(__name__)

def setup_lora(model: PreTrainedModel, lora_cfg: dict) -> PreTrainedModel:
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **lora_cfg)
    model = get_peft_model(model, lora_config)
    return model

def set_trainable_parameters(model: PreTrainedModel) -> PreTrainedModel:
    """Sets requires_grad=True for specific layers."""
    base_model_unwrapped = model.base_model.model
    if hasattr(base_model_unwrapped, "heat_embedding"):
        for param in base_model_unwrapped.heat_embedding.parameters():
             param.requires_grad = True
    if hasattr(base_model_unwrapped, "visual") and hasattr(base_model_unwrapped.visual, "blocks"):
        for block in base_model_unwrapped.visual.blocks:
            for module_name in ["attn_cross", "norm3", "norm4"]:
                module_obj = getattr(block, module_name, None)
                if module_obj:
                    for param in module_obj.parameters():
                        param.requires_grad = True
    model.print_trainable_parameters()
    return model

def initialize_custom_weights(model: PreTrainedModel):
     """Initializes weights for specific trainable layers."""
     base_model_unwrapped = model.base_model.model
     if hasattr(base_model_unwrapped, "heat_embedding"):
         init_transformer_block_weights(base_model_unwrapped.heat_embedding)
     if hasattr(base_model_unwrapped, "visual") and hasattr(base_model_unwrapped.visual, "blocks"):
        for i, block in enumerate(base_model_unwrapped.visual.blocks):
            for module_name in ["attn_cross", "norm3", "norm4"]:
                module_obj = getattr(block, module_name, None)
                if module_obj:
                     if list(module_obj.parameters()):
                         init_transformer_block_weights(module_obj)

class HeatmapInjectionExperiment(Experiment):
    eval_dataset: Dataset = None
    model: PreTrainedModel = None
    processor: ProcessorMixin = None
    data_collator: callable = None

    def prepare_model_structure(self) -> tuple[PreTrainedModel, ProcessorMixin]:
        """Prepares model structure: loads base model, applies LoRA config, sets custom layer trainability."""
        model = Qwen2_5_VLForConditionalGenerationWithHeatmap.from_pretrained(
            self.cfg.model.name,
            **self.cfg.model.kwargs
        )
        processor = AutoProcessor.from_pretrained(self.cfg.model.name)
        processor.tokenizer.padding_side = 'left'
        model = setup_lora(model, OmegaConf.to_object(self.cfg.lora))
        model = set_trainable_parameters(model)
        return model, processor

    def prepare_dataset(self):
        dataset = load_dataset(self.cfg.dataset.name)["train"]
        split_dataset = dataset.train_test_split(test_size=0.05)
        self.train_dataset = split_dataset["train"]
        self.eval_dataset = split_dataset["test"]

def main(config):
    experiment = HeatmapInjectionExperiment(config)

    trainer_kwargs = OmegaConf.to_container(experiment.cfg.trainer, resolve=True)
    training_args = TrainingArguments(**trainer_kwargs)

    resume_path_arg = training_args.resume_from_checkpoint
    actual_resume_path = None
    if isinstance(resume_path_arg, str) and os.path.isdir(resume_path_arg):
        actual_resume_path = resume_path_arg
    elif resume_path_arg is True:
        last_checkpoint = Trainer.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
             actual_resume_path = last_checkpoint

    experiment.model, experiment.processor = experiment.prepare_model_structure()
    experiment.processor.tokenizer.padding_side = 'left'

    if not actual_resume_path:
        initialize_custom_weights(experiment.model)
        training_args.resume_from_checkpoint = None # Explicitly set to None for fresh start
    else:
        training_args.resume_from_checkpoint = actual_resume_path # Pass the path to Trainer

    experiment.prepare_dataset()
    experiment.data_collator = DataCollator(experiment.processor, **experiment.cfg.dataset.kwargs)

    experiment.task_init()
    callbacks = [ClearMLCallback(experiment.task), SaveCustomWeightsCallback()]
    
    if actual_resume_path:
        custom_weights_path = _find_custom_weights_path(actual_resume_path)
        if custom_weights_path:
            _apply_custom_weights(experiment.model, custom_weights_path)


    trainer = Trainer(
        model=experiment.model,
        args=training_args,
        train_dataset=experiment.train_dataset,
        eval_dataset=experiment.eval_dataset,
        data_collator=experiment.data_collator,
        callbacks=callbacks,
    )
    trainer.train()

if __name__ == "__main__":
    fire.Fire(main)
