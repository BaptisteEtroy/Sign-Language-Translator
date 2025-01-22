import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models import ASLModel

###############################################################################
# 1. Loading the dataset and preprocessing it
###############################################################################
DATASET_PATH = "asl_alphabet_train"
BATCH_SIZE = 64
EPOCHS = 10
IMG_SIZE = 64

# Transform pipeline: Resize -> ToTensor -> Normalize
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load data using ImageFolder
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

# Split dataset: 80% training, 20% testing
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
print(f"Classes: {full_dataset.classes}")

###############################################################################
# 2. Training & Evaluation
###############################################################################
def main():
    # Device setup (CPU in this example)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize model
    num_classes = len(full_dataset.classes)  # e.g., 29 (A-Z + space/del/nothing)
    model = ASLModel(num_classes=num_classes).to(device)

    # Define loss (Cross Entropy) & optimizer (Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Save the model weights
    torch.save(model.state_dict(), "asl_model_pytorch.pth")
    print("Model saved to asl_model_pytorch.pth")

if __name__ == "__main__":
    main()