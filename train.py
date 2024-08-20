import torch
from data.dataset import VQADataset
from models.vilt_model import initialize_model
from transformers import ViltProcessor
from utils.utils import load_data

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    questions, annotations = load_data()
    dataset = VQADataset(questions, annotations, processor, '/path/to/images')
    model = initialize_model(processor.config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(50):
        for batch in dataset:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    train()
