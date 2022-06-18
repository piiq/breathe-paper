"""DenseNet NN object.

This file includes utilities and helper functions from the repository that contains
the PyTorch code for the following paper:
[Rethinking CNN Models for Audio Classification](https://arxiv.org/abs/2007.11154)

In the original project the classes and methods were located in different files and
folders so simple copying and pasting code from this section into the original
project might not work.
"""
from typing import Optional
import torch.nn as nn
import torchvision.models as models


class DenseNet(nn.Module):
    """
    DenseNet NN object.

    Inherits from torch.nn.Module
    """

    def __init__(self, dataset: str, pretrained: Optional[bool] = True):
        """Construct an NN instance.

        Parameters
        ----------
        dataset : str
            The dataset
        pretrained : bool, optional
            Flag if a pretrained model is required, by default True
        """
        super(DenseNet, self).__init__()
        num_classes = 50 if dataset == "ESC" else 10
        self.model = models.densenet201(pretrained=pretrained)
        self.model.classifier = nn.Linear(1920, num_classes)

    def forward(self, x):
        """Pre-configure forward function.

        The forward function is used to obtain the results of the model work.
        """
        output = self.model(x)
        return output
