import torch
import torch.optim as optim
import torch.nn as nn
from tqdm.notebook import tqdm
from models.anomaly_detection_model import AnomalyModel
from utils.data_setup import train_loader, valid_loader

def train_model(num_epochs=5, learning_rate=0.001, num_classes=5,save_path="anomaly-detection-efficientnet.pt"):
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = AnomalyModel(num_classes=num_classes)
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store loss values
    train_losses, val_losses = [], []

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training Loop"):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc='Training loop', leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc='Validation loop', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

        val_loss = running_loss / len(valid_loader.dataset)
        val_losses.append(val_loss)

        # Print stats
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), save_path)
    return model, train_losses, val_losses

if __name__ == '__main__':
    trained_model, train_losses, val_losses = train_model(num_epochs=5, learning_rate=0.001, num_classes=5)
