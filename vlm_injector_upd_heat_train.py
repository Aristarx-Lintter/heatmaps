import fire

from omegaconf import OmegaConf
import torch
from transformers import AutoProcessor, AutoConfig, PreTrainedModel, ProcessorMixin, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType

from src.common.dataset import DataCollator
from src.experiment import Experiment
from src.qwen2_5.fa_model import Qwen2_5_VLForConditionalGenerationWithHeatmap
from src.train_tools.callbacks import ClearMLCallback, SaveCustomWeightsCallback
from src.train_tools.initiator import init_transformer_block_weights

# torch.autograd.set_detect_anomaly(True)


def setup_lora(model: PreTrainedModel, lora_cfg: dict) -> PreTrainedModel:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        **lora_cfg 
    )
    model = get_peft_model(model, lora_config)
    return model


def set_trainable_parameters(model: PreTrainedModel) -> PreTrainedModel:
    for param in model.base_model.model.heat_embedding.parameters():
         param.requires_grad = True
    for param in model.base_model.model.visual.blocks[-1].parameters():
         param.requires_grad = True

    model.print_trainable_parameters()
    return model


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

        init_transformer_block_weights(model.visual.heat_blocks)
        init_transformer_block_weights(model.heat_embedding) # Uncomment if heat_embedding needs similar init

        model = setup_lora(model, OmegaConf.to_object(self.cfg.lora))
        model = set_trainable_parameters(model)

        return model, processor

    def prepare_dataset(self):
        dataset = load_dataset(self.cfg.dataset.name)["train"]
        split_dataset = dataset.train_test_split(test_size=0.05)
        self.train_dataset = split_dataset["train"]
        self.eval_dataset = split_dataset["test"]

    def prepare_for_training(self):
        self.model, self.processor = self.prepare_model()
        self.prepare_dataset()
        self.train_args = TrainingArguments(**self.cfg.trainer)
        self.data_collator = DataCollator(self.processor, **self.cfg.dataset.kwargs)


def main(config):
    # path = "Archistrax/Qwen2_5_VL"
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
