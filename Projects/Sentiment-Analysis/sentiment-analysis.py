# -----------------------------
# Sentiment Analysis using BERT (Manual Dataset)
# -----------------------------

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import random

# -----------------------------
# 1. Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2. Manual Dataset
# -----------------------------
train_texts = [
    "I love this movie, it was amazing!",        # Positive
    "What a fantastic experience",              # Positive
    "I hate this movie, it was terrible",       # Negative
    "This is the worst thing I have ever seen", # Negative
    "Absolutely wonderful!",                     # Positive
    "I do not like this at all",                 # Negative
    "The plot was fantastic and engaging",       # Positive
    "I feel disappointed after watching it",    # Negative
    "Brilliant acting and great storyline",     # Positive
    "I regret spending time on this movie",     # Negative
]

train_labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1=Positive, 0=Negative

# -----------------------------
# 3. Dataset class
# -----------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -----------------------------
# 4. Tokenizer and DataLoader
# -----------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = SentimentDataset(train_texts, train_labels, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------------
# 5. Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)
model = model.to(device)

# -----------------------------
# 6. Optimizer & Loss
# -----------------------------
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# 7. Training loop
# -----------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
    
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.2f}%")

# -----------------------------
# 8. Prediction function
# -----------------------------
def predict_sentiment(text):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, pred = torch.max(logits, dim=1)
    
    return "Positive" if pred.item() == 1 else "Negative"

# -----------------------------
# 9. Interactive testing
# -----------------------------
print("\n✅ Training done! You can now type a sentence to get sentiment prediction.")
while True:
    text = input("\nEnter a sentence (or type 'exit' to quit): ")
    if text.lower() == 'exit':
        break
    sentiment = predict_sentiment(text)
    print(f"Predicted Sentiment: {sentiment}")
