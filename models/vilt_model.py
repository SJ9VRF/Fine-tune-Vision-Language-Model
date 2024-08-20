# Defines the ViltForQuestionAnswering model

import torch
from transformers import ViltForQuestionAnswering

def initialize_model(config, device):
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                     id2label=config.id2label,
                                                     label2id=config.label2id)
    model.to(device)
    return model
