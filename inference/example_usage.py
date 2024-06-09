import torch
import torchvision.transforms as transforms
import numpy as np
from glob import glob
from utils.predict_utils import preprocess_image, predict, visualize_predictions
from evaluation.load_model import load_model
from utils.data_setup import train_dataset

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model_path = "./anomaly-detection-efficientnet.pt"
model = load_model(model_path, num_classes=5)
model.to(device)

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
