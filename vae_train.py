import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.qwen2_5.vae.heatmap import Qwen2_5_VisionTransformerWithHeatmap
from src.qwen2_5.vae.trainer import VAETrainingModule
from src.train_tools.lightning.loggers import ClearMLLightningLogger

load_dotenv()


if __name__ == "__main__":
    batch_size = 400

    vit = Qwen2_5_VisionTransformerWithHeatmap.from_pretrained(
        "./qwen2.5-vit",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    # dataset.push_to_hub("Archistrax/VIT_Qwen2_5_VL")
    dataset = load_dataset("Archistrax/VIT_Qwen2_5_VL")
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

    train_dataloader = DataLoader(dataset["train"].with_format("torch"), batch_size=batch_size, num_workers=12,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset["test"].with_format("torch"), batch_size=batch_size, num_workers=12,
                                 shuffle=False)

    clearml_logger = ClearMLLightningLogger(project_name="qwen2.5-vae", task_name="vae-phase1-lightning-one")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", save_top_k=1, mode="min", filename="vae-{epoch:02d}-{val_loss:.4f}"
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        callbacks=[checkpoint_callback],
        logger=clearml_logger,
        log_every_n_steps=10,
    )

    pl_module = VAETrainingModule(model=vit)
    trainer.fit(pl_module, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

