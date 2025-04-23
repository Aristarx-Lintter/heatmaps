import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from clearml import Task
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToPILImage
from PIL import Image

from src.qwen2_5.vae.heatmap import Qwen2_5_VisionTransformerWithHeatmap


path = "./Qwen2.5-VL-3B-Instruct"


class VAETrainingModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.post_merger_injector.to_latent.parameters():
            param.requires_grad = True
        for param in self.model.post_merger_injector.from_latent.parameters():
            param.requires_grad = True
        for param in self.model.post_merger_injector.heatmap_proj.parameters():
            param.requires_grad = False

    def forward(self, pixel_values, grid_thw, heatmap_flat):
        return self.model(pixel_values, grid_thw, heatmap_flat=heatmap_flat)

    def shared_step(self, batch):
        pixel_values = batch["pixel_values"].to(self.device)
        grid_thw = batch["image_grid_thw"].to(self.device)

        with torch.no_grad():
            hidden_states = self.model(pixel_values, grid_thw)
        merged_batch_size = hidden_states.shape[0]
        heatmap_flat = torch.zeros((merged_batch_size, 1), device=self.device)
        biased_hidden_states = self.model.post_merger_injector(hidden_states, heatmap_flat)
        loss = self.criterion(biased_hidden_states, hidden_states)
        self.log("loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4)


