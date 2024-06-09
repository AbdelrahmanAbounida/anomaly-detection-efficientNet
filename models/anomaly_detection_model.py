import torch.nn as nn
import timm

class AnomalyModel(nn.Module):
    def __init__(self, num_classes=5):
        super(AnomalyModel, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=True)
        efficient_net_out_size = 1280  # efficientnet output features size

        self.features = nn.Sequential(
            *list(self.model.children())[:-1]
        )  # remove last classification layer

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling layer

        self.dropout = nn.Dropout(0.5)  # Dropout layer with 0.5 dropout rate

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(efficient_net_out_size, num_classes),
            nn.Softmax(dim=1)  # Softmax activation layer
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        output = self.classifier(x)
        return output
