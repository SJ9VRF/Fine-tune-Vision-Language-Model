import json
from os import listdir
from os.path import isfile, join
from PIL import Image
import torch
from transformers import ViltProcessor

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, questions, annotations, processor, image_folder):
        self.questions = questions
        self.annotations = annotations
        self.processor = processor
        self.image_folder = image_folder

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        question = self.questions[idx]
        image_path = join(self.image_folder, f"COCO_val2014_{annotation['image_id']:012d}.jpg")
        image = Image.open(image_path)
        text = question['question']
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        labels = annotation['labels']
        scores = annotation['scores']
        targets = torch.zeros(len(self.processor.config.id2label))
        for label, score in zip(labels, scores):
            targets[label] = score
        encoding['labels'] = targets
        return encoding
