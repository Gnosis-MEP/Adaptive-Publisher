import os

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from adaptive_publisher.models.base import BaseModel
from adaptive_publisher.conf import (
    MODELS_PATH,
    CLS_MODEL_ID,
)


class OIClsModel(BaseModel):
    def __init__(self, oi_cls_index=1):
        super().__init__()
        self.oi_cls_index = oi_cls_index
        self.setup()

    def setup(self):
        num_classes = 2
        base_model = self.get_base_fine_tuned_model(models.mobilenet_v3_large(), num_classes, freeze=False)

        model_path = os.path.join(MODELS_PATH, f'{CLS_MODEL_ID}.pth')
        self.device = torch.device('cpu')

        # Load the saved state dictionary
        base_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = base_model.to(self.device)
        self.model.eval()

    def get_base_fine_tuned_model(self, base_model, num_classes=2, freeze=True):
        num_classes = 2
        base_model.classifier[-1] = nn.Linear(base_model.classifier[-1].in_features, num_classes)
        return base_model

    def predict(self, new_image_frame, last_key_frame=None):
        input_batch = new_image_frame.unsqueeze(0)
        prediciton = self.model(input_batch).squeeze(0).softmax(0)
        class_probs = prediciton.tolist()
        return class_probs[self.oi_cls_index]