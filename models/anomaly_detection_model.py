import torch.nn as nn
import timm

class AnomalyModel(nn.Module):
    def __init__(self, num_classes=5):
        super(AnomalyModel, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=True)
        efficient_net_out_size = self.model.classifier.in_features  # Dynamically get the output features size

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(efficient_net_out_size, num_classes)
        )

        # Remove the last classification layer of the pretrained model
        self.features = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        x = self.features(x)  # Use EfficientNet layers and calculate them first
        output = self.classifier(x)  # This is the last layer which takes 1280 and returns the number of classes
        return output
