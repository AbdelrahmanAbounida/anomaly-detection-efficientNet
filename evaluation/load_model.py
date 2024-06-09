from models.anomaly_detection_model import AnomalyModel
import torch


def load_model(model_path, num_classes=5):
    """
    Loads the model from the specified path.

    Parameters:
    - model_path (str): Path to the model's state dictionary.
    - num_classes (int): Number of classes for the classifier.

    Returns:
    - model (nn.Module): The loaded model.
    """
    model = AnomalyModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))

    if torch.cuda.is_available():
        model.cuda()

    return model
