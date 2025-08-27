import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# -----------------------------
# 1. Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -----------------------------
# 2. Data preparation
# -----------------------------
data_dir = ""  # path to Kaggle dataset root

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # both ResNet/VGG expect 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Imagenet mean
                         [0.229, 0.224, 0.225]) # Imagenet std
])

full_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)

# train/val split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# -----------------------------
# 3. Model loader
# -----------------------------
def get_model(name="resnet18", num_classes=2, freeze_backbone=True):
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
    
    return model.to(device)

# -----------------------------
# 4. Training function
# -----------------------------
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, save_name="model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
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
    print(f"Model saved as {save_name}")

# -----------------------------
# 5. Run Training
# -----------------------------
# Train ResNet18
resnet_model = get_model("resnet18", freeze_backbone=True)
train_model(resnet_model, train_loader, val_loader, epochs=5, save_name="cats_vs_dogs_resnet18.pth")

# Train VGG16
vgg_model = get_model("vgg16", freeze_backbone=True)
train_model(vgg_model, train_loader, val_loader, epochs=5, save_name="cats_vs_dogs_vgg16.pth")
