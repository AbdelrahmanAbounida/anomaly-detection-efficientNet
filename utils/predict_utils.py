import matplotlib.pyplot as plt
from PIL import Image
import torch

# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

# Visualization
def visualize_predictions(original_image): #, probabilities, class_names
    fig, axarr = plt.subplots(1, 1, figsize=(3, 3))

    # Display image
    axarr.imshow(original_image)
    axarr.axis("off")

    plt.tight_layout()
    plt.show()
