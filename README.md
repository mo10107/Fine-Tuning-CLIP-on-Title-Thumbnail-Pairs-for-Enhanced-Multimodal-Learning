# Fine-Tuning CLIP on Title-Thumbnail Pairs for Enhanced Multimodal Learning

## Overview
This project fine-tunes OpenAI's CLIP model on YouTube title-thumbnail pairs to improve multimodal learning and retrieval capabilities. The model learns to associate video thumbnails with their corresponding titles, optimizing for better image-text alignment and retrieval.

## Features
- **Pretrained CLIP Model**: Utilizes `sentence-transformers/clip-ViT-L-14` for training.
- **Custom Training**: Selectively fine-tunes specific layers to retain generalization while improving task-specific performance.
- **Dataset**: Uses `shawhin/yt-title-thumbnail-pairs` dataset.
- **Triplet Evaluation**: Implements `TripletEvaluator` to measure performance.
- **Recall@K Evaluation**: Introduces an `ImageTextRetrievalEvaluator` to assess image-text matching accuracy.
- **Efficient Training**: Employs `MultipleNegativesRankingLoss` to enhance contrastive learning.

## Installation
To run this project, install the required dependencies:
```bash
pip install torch transformers datasets sentence-transformers
```

## Dataset
The dataset is loaded using the `datasets` library:
```python
from datasets import load_dataset
dataset = load_dataset("shawhin/yt-title-thumbnail-pairs")
```

## Model Fine-Tuning
The fine-tuning process involves freezing most CLIP layers and training specific projection layers:
```python
trainable_layers_list = ['projection']
for name, param in model.named_parameters():
    param.requires_grad = False
    if any(layer in name for layer in trainable_layers_list):
        param.requires_grad = True
```

## Data Preprocessing
Prepares the dataset by fetching images and mapping text labels:
```python
def preprocess(batch):
    image_list = [Image.open(requests.get(url, stream=True).raw) for url in batch["thumbnail_url"]]
    return {
        "anchor": image_list,
        "positive": batch["title"],
        "negative": batch["title_neg"]
    }

dataset = dataset.map(preprocess, batched=True)
```

## Training
Training is performed using `SentenceTransformerTrainer` with hyperparameter tuning:
```python
trainer = SentenceTransformerTrainer(
    model=model,
    args=SentenceTransformerTrainingArguments(
        output_dir="models/clip-title-thumbnail-embeddings",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        learning_rate=1e-4,
        eval_strategy="epoch",
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    loss=MultipleNegativesRankingLoss(model),
    evaluator=[evaluator_recall_train, evaluator_recall_valid],
)
trainer.train()
```

## Evaluation
After training, the model is evaluated using triplet and recall-based metrics:
```python
evaluator_test = create_triplet_evaluator("test")
evaluator_recall_test = create_recall_evaluator("test")
print("Test Triplet Accuracy:", evaluator_test(model))
print("Test Recall@1:", evaluator_recall_test(model))
```

## Results
The model is evaluated on its ability to correctly retrieve corresponding text-image pairs, measured via:
- **Triplet Accuracy**: Checks if the model correctly associates images with their matching titles.
- **Recall@K**: Measures how often the correct title appears in the top-K predictions.

