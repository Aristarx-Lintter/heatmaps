import torch.nn.functional as F
from torchvision import transforms


class ResizeTensor:
    """
    Простой класс-трансформ, который повторяет поведение F.interpolate(..., align_corners=False)
    для тензора формата (C, H, W).
    """

    def __init__(self, size=(14, 14), mode='bilinear', align_corners=False):
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, tensor):
        tensor = tensor.unsqueeze(0)  # -> (1, C, H, W)
        tensor = F.interpolate(
            tensor, size=self.size,
            mode=self.mode,
            align_corners=self.align_corners
        )  # -> (1, C, newH, newW)
        return tensor.squeeze(0)  # -> (C, newH, newW)


def get_heatmap_transformation(h, w):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("L")),
        transforms.ToTensor(),  # теперь тензор формата (1, H, W)
        transforms.Lambda(lambda t: (t - t.mean()) / (t.std() + 1e-8)),
        ResizeTensor(size=(int(h), int(w)), mode='bilinear', align_corners=False),
        transforms.Lambda(lambda t: t.flatten())
    ])
