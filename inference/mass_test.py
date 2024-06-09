from utils.predict_utils import preprocess_image, predict, visualize_predictions
from models.anomaly_detection_model import AnomalyModel
import torchvision.transforms as transforms
from utils.data_setup import train_dataset
from glob import glob
import numpy as np
import torch

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model
model = AnomalyModel(num_classes=5)
model.to(device)

# Load pre-trained model weights
model.load_state_dict(torch.load("../data/model_weights.pth"))

# Transformation for the test images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Select random test images
test_images = glob('./data/processed/test/*/*')
test_examples = np.random.choice(test_images, 10)

# Predict and visualize
for example in test_examples:
    original_image, image_tensor = preprocess_image(example, transform)
    probabilities = predict(model, image_tensor, device)

    class_names = train_dataset.classes
    predicted_class = class_names[np.argmax(probabilities)]
    print(f"Predicted class: {predicted_class}")
    visualize_predictions(original_image) #, probabilities, class_names
