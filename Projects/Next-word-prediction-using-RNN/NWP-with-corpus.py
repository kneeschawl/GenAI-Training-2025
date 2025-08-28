# manual_corpus_rnn.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os

# -----------------------------
# 1. Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2. Load manual corpus
# -----------------------------
corpus_file = r"C:\Users\Dell\OneDrive\Documents\GenAI Training\Projects\Next-word-prediction-using-RNN\manual_corpus.txt" # Provide your large text corpus here
if not os.path.exists(corpus_file):
    raise FileNotFoundError(f"{corpus_file} not found. Please provide a corpus.")

with open(corpus_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip() != ""]

# Split into training and validation
train_texts = lines[:int(len(lines)*0.8)]
valid_texts = lines[int(len(lines)*0.8):]

print(f"Training lines: {len(train_texts)}, Validation lines: {len(valid_texts)}")

# -----------------------------
# 3. Build vocabulary (top 20k words)
# -----------------------------
all_text = " ".join(train_texts).lower()
words = all_text.split()
counter = Counter(words)
top_words = [w for w,_ in counter.most_common(20000)]
word2idx = {w: i+1 for i, w in enumerate(top_words)}  # 0 reserved for <UNK>
idx2word = {i+1: w for i, w in enumerate(top_words)}
vocab_size = len(word2idx) + 1
print(f"Vocabulary size: {vocab_size}")

# -----------------------------
# 4. Dataset class
# -----------------------------
seq_length = 5

class WordDataset(Dataset):
    def __init__(self, texts, seq_length):
        self.texts = texts
        self.seq_length = seq_length
    
    def __len__(self):
        return sum(max(len(t.split()) - self.seq_length, 0) for t in self.texts)
    
    def __getitem__(self, idx):
        count = 0
        for line in self.texts:
            words_line = line.lower().split()
            if len(words_line) <= self.seq_length:
                continue
            for i in range(self.seq_length, len(words_line)):
                if count == idx:
                    seq = words_line[i-self.seq_length:i]
                    target = words_line[i]
                    seq_idx = [word2idx.get(w, 0) for w in seq]
                    target_idx = word2idx.get(target, 0)
                    return torch.tensor(seq_idx), torch.tensor(target_idx)
                count += 1
        return torch.zeros(self.seq_length, dtype=torch.long), torch.tensor(0)

train_loader = DataLoader(WordDataset(train_texts, seq_length), batch_size=32, shuffle=True)
valid_loader = DataLoader(WordDataset(valid_texts, seq_length), batch_size=32, shuffle=False)

# -----------------------------
# 5. RNN Model
# -----------------------------
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

model_file = "manual_rnn_model.pth"
model = WordRNN(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 6. Load saved model if exists
# -----------------------------
try:
    model.load_state_dict(torch.load(model_file))
    print("Loaded saved model. Skipping training.")
except FileNotFoundError:
    print("No saved model found. Training now...")
    epochs = 1  # Adjust if you want longer training
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seqs, targets in valid_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                outputs, _ = model(seqs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    torch.save(model.state_dict(), model_file)
    print(f"Model saved as {model_file}")

# -----------------------------
# 7. Interactive prediction
# -----------------------------
def predict_next_words(model, seed_text, n_words=5, temperature=1.0):
    model.eval()
    words = seed_text.lower().split()
    for _ in range(n_words):
        seq = words[-seq_length:]
        seq_idx = torch.tensor([[word2idx.get(w, 0) for w in seq]]).to(device)
        with torch.no_grad():
            output, _ = model(seq_idx)
            probs = output.squeeze().div(temperature).exp()
            word_idx = torch.multinomial(probs, 1).item()
        words.append(idx2word.get(word_idx, "<UNK>"))
    return " ".join(words)

print("\n--- Manual Corpus Next-word Prediction Demo ---")
while True:
    seed = input("Enter seed text (or 'exit' to quit): ")
    if seed.lower() == "exit":
        break
    temp = input("Enter temperature (0.7=conservative,1.0=normal,1.2=creative): ")
    try:
        temp = float(temp)
    except:
        temp = 1.0
    predicted = predict_next_words(model, seed, n_words=5, temperature=temp)
    print("Predicted text:", predicted)
    print("-----------------------------------")
