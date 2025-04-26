import fire

import torch
from transformers import AutoProcessor, AutoConfig, PreTrainedModel, ProcessorMixin, TrainingArguments, Trainer
from datasets import Dataset, load_dataset

from src.common.dataset import DataCollator
from src.experiment import Experiment
from src.qwen2_5.fa_model import Qwen2_5_VLForConditionalGenerationWithHeatmap
from src.train_tools.callbacks import ClearMLCallback
from src.train_tools.initiator import init_transformer_block_weights

# torch.autograd.set_detect_anomaly(True)


class HeatmapInjectionExperiment(Experiment):
    eval_dataset: Dataset = None
    model: PreTrainedModel = None
    processor: ProcessorMixin = None
    train_args: TrainingArguments = None
    data_collator: callable = None

    def prepare_model(self) -> tuple[PreTrainedModel, ProcessorMixin]:

        # hf_config = AutoConfig.from_pretrained(self.cfg.model.name, trust_remote_code=True)
        # hf_config.vision_config.latent_dim = 512

        model = Qwen2_5_VLForConditionalGenerationWithHeatmap.from_pretrained(
            self.cfg.model.name,
            # config=hf_config,
            # ignore_mismatched_sizes=True,
            **self.cfg.model.kwargs
        )
        processor = AutoProcessor.from_pretrained(self.cfg.model.name)
        processor.tokenizer.padding_side = 'left'

        for param in model.parameters():
            param.requires_grad = False

        init_transformer_block_weights(model.visual.blocks[-1])
        # init_transformer_block_weights(model.heat_embedding)

        for param in model.visual.blocks[-1].parameters():
            param.requires_grad = True
        for param in model.heat_embedding.parameters():
            param.requires_grad = True

        return model, processor

    def prepare_dataset(self):
        dataset = load_dataset(self.cfg.dataset.name)["train"]
        length = len(dataset)
        self.train_dataset = dataset.select(range(int(length * 0.95)))
        self.eval_dataset = dataset.select(range(int(length * 0.95), length))

    def prepare_for_training(self):
        self.model, self.processor = self.prepare_model()
        self.prepare_dataset()
        self.train_args = TrainingArguments(**self.cfg.trainer)
        self.data_collator = DataCollator(self.processor, self.cfg.dataset.transcribation_feature_name)


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
        callbacks=[ClearMLCallback(experiment.task)]
    )

    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
