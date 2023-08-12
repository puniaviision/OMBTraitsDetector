from torchvision import models
import torch

resnet = models.resnet50(pretrained=True)
resnet.eval()