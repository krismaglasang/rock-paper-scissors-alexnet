import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_filters, p=0.0) -> None:
        super(CNN, self).__init__()
        self.n_filters = n_filters
        self.p = p

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=n_filters,
            kernel_size=3
        )

        self.conv2 = nn.Conv2d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=3
        )

        self.fc1 = nn.Linear(
            in_features=n_filters*5*5,
            out_features=50
        )

        self.fc2 = nn.Linear(
            in_features=50,
            out_features=3
        )

        self.drop = nn.Dropout2d(p=self.p)
    
    def featurizer(self, x):
        # 1st convolutional block
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # 2nd convolutional block
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Flatten in prep for the neural network
        return nn.Flatten()(x)

    def classifier(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.drop(x)
        x = self.fc2(x)
        
        # Logits as output
        return x

    def forward(self, x):
        x = self.featurizer(x)
        x = self.classifier(x)
        return x


    