import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image
import gradio as gr

# -----------------------------
# 1. Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2. Data preparation
# -----------------------------
data_dir = r"C:\Users\Dell\OneDrive\Documents\GenAI Training\Projects\CNN-for-Image-Classification\cats_dogs"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("📂 Loading dataset from:", data_dir)
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = full_dataset.classes
print("Found classes:", class_names)
print("Total images:", len(full_dataset))

# -----------------------------
# 2a. Quick test: small subset
# -----------------------------
small_dataset_size = 1000
if small_dataset_size < len(full_dataset):
    indices = random.sample(range(len(full_dataset)), small_dataset_size)
    dataset = torch.utils.data.Subset(full_dataset, indices)
else:
    dataset = full_dataset

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# -----------------------------
# 3. Model loader
# -----------------------------
def get_model(name="resnet18", num_classes=2, freeze_backbone=True):
    print(f"⬇️ Downloading pretrained {name} weights (if not cached)...")
    if name == "resnet18":
        model = models.resnet18(pretrained=True)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "vgg16":
        model = models.vgg16(pretrained=True)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Choose 'resnet18' or 'vgg16'")
    print("✅ Model ready!")
    return model.to(device)

# -----------------------------
# 4. Training function
# -----------------------------
def train_model(model, train_loader, val_loader, epochs=2, lr=0.001, save_name="model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    print("\n🚀 Training started...\n")
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), save_name)
    print(f"\n💾 Model saved as {save_name}")

# -----------------------------
# 5. Prediction function (works for file paths & Gradio uploads)
# -----------------------------
def predict_image_gradio(image_or_path):
    """
    Handles both:
    - File path string (for terminal tests)
    - Uploaded images from Gradio (NumPy array)
    """
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path).convert("RGB")
    elif not isinstance(image_or_path, Image.Image):
        image = Image.fromarray(image_or_path).convert("RGB")
    else:
        image = image_or_path.convert("RGB")
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    resnet_model.eval()
    with torch.no_grad():
        outputs = resnet_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# -----------------------------
# 6. Main
# -----------------------------
if __name__ == "__main__":
    # Train ResNet18 quickly
    resnet_model = get_model("resnet18", freeze_backbone=True)
    train_model(resnet_model, train_loader, val_loader, epochs=2, save_name="quick_test_resnet18.pth")

    # ---- Random sample test from validation set ----
    rand_idx = random.randint(0, len(val_dataset)-1)
    img, label = val_dataset[rand_idx]
    img_path, _ = full_dataset.samples[val_dataset.indices[rand_idx]]
    prediction = predict_image_gradio(img_path)
    print(f"\n🔍 Random validation sample: {os.path.basename(img_path)}")
    print(f"Ground Truth: {class_names[label]} | Predicted: {prediction}")

    # ---- Launch Gradio web app ----
    print("\n🌐 Launching Gradio web app...")
    gr.Interface(fn=predict_image_gradio, inputs="image", outputs="text",
                 title="Cat vs Dog Classifier",
                 description="Upload any image (jpg/png) and the model will predict Cat or Dog.").launch()
