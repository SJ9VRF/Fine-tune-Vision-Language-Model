# Fine-tune-Vision-Language-Model

This repository contains the implementation of the Vision-and-Language Transformer (ViLT) model fine-tuned for Visual Question Answering (VQA) tasks. The project is structured to be easy to set up and use, providing a streamlined approach for experimenting with different configurations and datasets.


![Screenshot_2024-08-08_at_9 50 25_PM-removebg-preview](https://github.com/user-attachments/assets/9d82bbb9-814b-4336-bf7b-efba7e19b8d9)



## Installation

1. **Clone the Repository**
```bash
git clone https://your-repository-url.git
cd vilt-vqa
```


2. **Install Dependencies**
```bash
pip install -r requirements.txt
```


3. **Download Data**
Ensure that your data files are in the `data/` directory as specified in `settings.py`.

## Usage

### Training the Model

To train the model, run:
```bash
python train.py
```

This script will train the model using the configurations specified in `config/settings.py`.

### Making Predictions

To perform inference with a pre-trained model, run:

```bash
python infer.py --image_path 'path/to/image.jpg' --question 'What is in the picture?'
```

This will load the trained model and output the top predictions for the specified image and question.

## Configuration

Edit `config/settings.py` to modify paths, model parameters, and other settings like device configuration for GPU acceleration.



