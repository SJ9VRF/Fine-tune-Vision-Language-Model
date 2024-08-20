# Script for inference

import torch
from PIL import Image
from transformers import ViltProcessor
from models.vilt_model import initialize_model
from utils.utils import id_from_filename, load_data
import numpy as np

def load_model(device):
    """
    Load the trained model and processor.
    """
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    model = initialize_model(processor.config, device)
    model.load_state_dict(torch.load("path_to_trained_model.pth"))
    model.eval()
    return model, processor

def predict(image_path, question, model, processor, device):
    """
    Make a prediction using the model for a single image and question.
    """
    image = Image.open(image_path).convert("RGB")
    encoding = processor(image, question, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
    
    return probs.squeeze()

def get_top_answers(probs, processor, top_k=5):
    """
    Retrieve the top K answers based on the model's output probabilities.
    """
    probs, indices = torch.topk(probs, top_k)
    id2label = processor.config.id2label
    return [(prob.item(), id2label[idx.item()]) for prob, idx in zip(probs, indices)]

def main(image_path, question):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(device)
    
    probs = predict(image_path, question, model, processor, device)
    top_answers = get_top_answers(probs, processor)
    
    print(f"Question: {question}")
    print("Predicted Answers:")
    for prob, answer in top_answers:
        print(f"{answer}: {prob:.4f}")

if __name__ == "__main__":
    # Example usage
    main("path_to_image.jpg", "What is in the picture?")
