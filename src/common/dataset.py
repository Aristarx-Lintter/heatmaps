from typing import List, Dict, Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import PreTrainedTokenizer

from src.common.templates import messages_template, answer_template
from src.common.transforms import get_heatmap_transformation


def find_substring(input_ids: torch.Tensor, ref_ids: List[int]):
    start_index = -1
    for i in range(len(input_ids) - len(ref_ids) + 1):
        if input_ids[i: i + len(ref_ids)].tolist() == ref_ids:
            start_index = i
            break
    if start_index == -1:
        raise ValueError("Target sequence not found.")
    end_index = start_index + len(ref_ids)
    return start_index, end_index


def create_labels(input_ids: torch.Tensor, answers: List[str], tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    """
    Create labels for SFT training. It masks all tokens after the start token with excluding_probability
    and after end token for the rest.
    Args:
        input_ids: ids from tokenizer output
        ....

    Returns: tensor with masks  for each input_ids
    """

    labels = torch.full_like(input_ids, fill_value=-100)

    for i, row in enumerate(input_ids):
        start_index, end_index = find_substring(
            row, tokenizer(answers[i], add_special_tokens=False)["input_ids"]
        )
        labels[i, start_index:end_index] = row[start_index:end_index]

    return labels


def process_injection(image_grid_thw, features):
    heatmap_flat = []
    for thw, feature in zip(image_grid_thw, features):
        _, h, w = thw
        transformation = get_heatmap_transformation(h, w)
        heatmap_flat.append(transformation(feature["heatmap"]).unsqueeze(1))

    return torch.stack(heatmap_flat)


class DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def data_collator(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            return {}

        messages = []
        answers = []
        for feature in features:
            messages.append(messages_template(feature["image"], feature["transcribation"]))
            answers.append(answer_template.format(ans_text=feature["transcribation"]))

        texts = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)
        batch = self.processor(
            text=texts, images=image_inputs, padding=True, return_tensors="pt"
        )  # ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']

        batch["labels"] = create_labels(batch["input_ids"], answers, self.processor.tokenizer)
        batch["heatmap_flat"] = process_injection(batch["image_grid_thw"], features)

        return batch
