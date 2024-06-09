from utils.data_setup import train_dataset, test_dataset
from models.anomaly_detection_model import AnomalyModel
from evaluation.analyze_results import analyze_results
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from training.train import train_model
import argparse
import logging
import torch

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    # Training the model
    logger.info("Training the model...")
    trained_model, train_losses, val_losses = train_model(num_epochs=5, learning_rate=0.001, num_classes=5, save_path="anomaly-detection-efficientnet.pt")
    logger.info("Model training completed.")

def evaluate():
    # Load the trained model
    model_path = "anomaly-detection-efficientnet.pt"
    logger.info(f"Loading the trained model from '{model_path}'...")
    loaded_model = AnomalyModel(num_classes=5)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.to(device)
    logger.info("Model loaded successfully.")

    # Transformation for the test images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Perform inference on test data
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    true_labels = []
    predictions = []
    probabilities_list = []

    logger.info("Performing inference on test data...")
    for images, labels in test_loader:
        images = images.to(device)
        outputs = loaded_model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten()
        predicted_class = int(torch.argmax(outputs))
        true_class = int(labels)
        true_labels.append(true_class)
        predictions.append(predicted_class)
        probabilities_list.append(probabilities)
    
    logger.info("Inference completed.")

    # Analyze the results
    class_names = train_dataset.classes
    logger.info("Analyzing the results...")
    analyze_results(true_labels, predictions, class_names, probabilities_list)
    logger.info("Result analysis completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection Model for 3D Printing Process")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")

    args = parser.parse_args()

    if args.train:
        train()
    elif args.evaluate:
        evaluate()
    else:
        logger.error("Please specify either '--train' or '--evaluate' argument.")
