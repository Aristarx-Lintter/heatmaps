import fire
import re

from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoConfig, PreTrainedModel, ProcessorMixin, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType

from src.common.dataset import DataCollator
from src.experiment import Experiment
from src.qwen2_5.fa_model import Qwen2_5_VLForConditionalGenerationWithHeatmap
from src.train_tools.callbacks import ClearMLCallback, SaveCustomWeightsCallback

# torch.autograd.set_detect_anomaly(True)


def setup_lora(model: PreTrainedModel, lora_cfg: dict) -> PreTrainedModel:
    target_modules_patterns = lora_cfg.pop('target_modules_patterns', [])

    all_module_names = [name for name, _ in model.named_modules()]
    final_target_modules = set()

    if target_modules_patterns:
        print("\n=== Applying LoRA to modules based on patterns: ===")
        for pattern_str in target_modules_patterns:
            try:
                pattern = re.compile(pattern_str)
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{pattern_str}': {e}. Skipping this pattern.")
                continue
            
            found_for_pattern = False
            for module_name in all_module_names:
                if pattern.fullmatch(module_name):
                    final_target_modules.add(module_name)
                    if not found_for_pattern:
                        print(f"  Pattern '{pattern_str}' matched:")
                        found_for_pattern = True
                    print(f"    - {module_name}")
            if not found_for_pattern:
                print(f"  Pattern '{pattern_str}' did not match any module names.")
        print("==================================================")
    else:
        print("Warning: No target_modules_patterns specified in LoRA config.")

    if not final_target_modules:
        print("Warning: No modules were matched by the provided patterns for LoRA. LoRA will not be applied to any specific layers directly via target_modules. Check your patterns.")

    lora_config_params = {
        **lora_cfg,
        'task_type': TaskType.CAUSAL_LM,
        'target_modules': list(final_target_modules) if final_target_modules else None,
    }

    lora_config = LoraConfig(**lora_config_params)
    
    model = get_peft_model(model, lora_config)
    print("\n=== LoRA Model Trainable Parameters (after get_peft_model) ===")
    model.print_trainable_parameters()
    print("===========================================================")
    return model


# def set_trainable_parameters(model: PreTrainedModel) -> PreTrainedModel:
#     for param in model.base_model.model.heat_embedding.parameters():
#          param.requires_grad = True
#     for param in model.base_model.model.visual.blocks[-1].parameters():
#          param.requires_grad = True

#     model.print_trainable_parameters()
#     return model


class HeatmapInjectionExperiment(Experiment):
    eval_dataset: Dataset = None
    model: PreTrainedModel = None
    processor: ProcessorMixin = None
    train_args: TrainingArguments = None
    data_collator: callable = None

    def prepare_model(self) -> tuple[PreTrainedModel, ProcessorMixin]:
        model = Qwen2_5_VLForConditionalGenerationWithHeatmap.from_pretrained(
            self.cfg.model.name,
            **self.cfg.model.kwargs
        )
        processor = AutoProcessor.from_pretrained(self.cfg.model.name)
        processor.tokenizer.padding_side = 'left'

        model = setup_lora(model, OmegaConf.to_object(self.cfg.lora))

        return model, processor

    def prepare_dataset(self):
        dataset = load_dataset(self.cfg.dataset.name)["train"]
        split_dataset = dataset.train_test_split(test_size=0.08)
        self.train_dataset = split_dataset["train"]
        self.eval_dataset = split_dataset["test"]

    def prepare_for_training(self):
        self.model, self.processor = self.prepare_model()
        self.prepare_dataset()
        self.train_args = TrainingArguments(**self.cfg.trainer)
        self.data_collator = DataCollator(self.processor, **self.cfg.dataset.kwargs)


def main(config):
    experiment = HeatmapInjectionExperiment(config)
    experiment.prepare_for_training()
    experiment.task_init()

    trainer = Trainer(
        model=experiment.model,
        processing_class=experiment.processor,
        train_dataset=experiment.train_dataset,
        eval_dataset=experiment.eval_dataset,
        data_collator=experiment.data_collator,
        args=experiment.train_args,
        callbacks=[ClearMLCallback(experiment.task), SaveCustomWeightsCallback()]
    )

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
