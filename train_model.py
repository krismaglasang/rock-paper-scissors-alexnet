import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from torchvision.datasets import ImageFolder
from torchvision.models import alexnet

from trainer import Trainer

torch.manual_seed(42)
alex = alexnet(pretrained=True)

'''
AlexNet Architecture

AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
'''

# replacing the last layer of the classifier with an Identity layer since
# we want to freeze the rest of the network
alex.classifier[6] = nn.Identity()

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False

freeze_layers(alex)


# Data preparation
normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])

composer = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    normalizer
])

train_data = ImageFolder('./rps', transform=composer)
val_data = ImageFolder('./rps-test-set', transform=composer)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_data,
    batch_size=16
)

# 1 forward pass thru the frozen layers to get the output features as a dataset
def preprocessed_dataset(model, loader):
    features = None
    labels = None

    for i, (x, y) in enumerate(loader):
      model.eval()
      output = model(x)
      if i == 0:
          features = output.detach().cpu()
          labels = y.cpu()
      else:
          features = torch.cat([features, output.detach().cpu()])
          labels = torch.cat([labels, y.cpu()])

    return TensorDataset(features, labels)

preprocessed_train_data = preprocessed_dataset(alex, train_loader)
preprocessed_val_data =  preprocessed_dataset(alex, val_loader)


# Create the dataloaders again using the output features from the frozen layers
preprocessed_train_loader = DataLoader(
    dataset=preprocessed_train_data,
    batch_size=16,
    shuffle=True
)

preprocessed_val_loader = DataLoader(
    dataset=preprocessed_val_data,
    batch_size=16
)

top_layer = nn.Sequential(nn.Linear(
    in_features=4096,
    out_features=3
))

optimizer = optim.Adam(
    top_layer.parameters(), lr=3e-4
)

loss_fn = nn.CrossEntropyLoss()

trainer = Trainer(top_layer, optimizer, loss_fn)
trainer.set_loaders(preprocessed_train_loader, preprocessed_val_loader)
trainer.train(10) # train for 10 epochs

'''
Epoch 1: 100%|██████████████████████████████| 24/24 [00:00<00:00, 3069.38it/s, accuracy=96.2, loss=0.159]
Epoch 2: 100%|██████████████████████████████| 24/24 [00:00<00:00, 4487.69it/s, accuracy=96.2, loss=0.143]
Epoch 3: 100%|████████████████████████████████| 24/24 [00:00<00:00, 4501.13it/s, accuracy=96, loss=0.176]
Epoch 4: 100%|██████████████████████████████| 24/24 [00:00<00:00, 4884.20it/s, accuracy=96.2, loss=0.142]
Epoch 5: 100%|██████████████████████████████| 24/24 [00:00<00:00, 4744.91it/s, accuracy=96.2, loss=0.144]
Epoch 6: 100%|████████████████████████████████| 24/24 [00:00<00:00, 5092.49it/s, accuracy=96, loss=0.173]
Epoch 7: 100%|████████████████████████████████| 24/24 [00:00<00:00, 4827.05it/s, accuracy=96, loss=0.171]
Epoch 8: 100%|████████████████████████████████| 24/24 [00:00<00:00, 4799.20it/s, accuracy=96, loss=0.177]
Epoch 9: 100%|████████████████████████████████| 24/24 [00:00<00:00, 4916.64it/s, accuracy=96, loss=0.185]
Epoch 10: 100%|███████████████████████████████| 24/24 [00:00<00:00, 4874.74it/s, accuracy=96, loss=0.165]
'''

# Attach the trained top layer to AlexNet
alex.classifier[6] = trainer.model

'''
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Sequential(
      (0): Linear(in_features=4096, out_features=3, bias=True)
    )
  )
)
'''
# Save the model
torch.save(alex.state_dict(), './model.pth')