import torch
from torchvision.datasets import ImageFolder

import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

class Trainer():
    def __init__(self, model, optimizer, loss_fn) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.classes = {}
        self.train_losses = []
        self.val_losses = []

    def plot_losses(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def set_loaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

        # simultaneously save the classes and their index
        if isinstance(val_loader.dataset, ImageFolder):
            self.classes = val_loader.dataset.class_to_idx

    def train(self, n_epochs):
        self.set_seed()

        for epoch in range(n_epochs):
            train_batch_loss = []
            val_batch_loss = []
            total_test = 0
            correct_test = 0

            for x, y in self.train_loader:
                yhat = self.model(x)
                loss = self.loss_fn(yhat, y)

                train_batch_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

                if self.scheduler != None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
        
            with torch.no_grad():
                with tqdm(self.val_loader, desc=f'Epoch {epoch + 1}', leave=True) as tepoch:
                    for x_val_batch, y_val_batch in tepoch:
                        self.model.eval()
                        yhat = self.model(x_val_batch)
                        loss = self.loss_fn(yhat, y_val_batch)

                        val_batch_loss.append(loss.item())

                        # Compute validation accuracy
                        _, predicted = torch.max(yhat.data, 1)
                        total_test += y_val_batch.size(dim=0)
                        correct_test += (predicted == y_val_batch).sum().item()

                        # It looks like in spite of this line being inside a for-loop,
                        # it is only run after all the batches have run
                        tepoch.set_postfix(loss=np.mean(val_batch_loss), accuracy=100*correct_test/total_test)

            self.train_losses.append(np.mean(train_batch_loss))
            self.val_losses.append(np.mean(val_batch_loss))

    def predict(self, x):
        self.model.eval()
        prediction = None
        # need to add the batch dimension
        # we use argmax to get the index of the max value
        pred_idx = torch.argmax(self.model(x.unsqueeze(0)), dim=1).item()
        for idx, _class in enumerate(self.classes):
            if idx == pred_idx:
                prediction = _class
                break
        return prediction
    
    def correct(self, x, y, threshold=0.5):
        self.model.eval()
        yhat = self.model(x)
        y = y
        self.model.train()

        n_samples, n_dims = yhat.shape
        if n_dims > 1:
            # In a multiclass classification, the largest logit
            # always wins, so we don't bother getting probabilities

            # This is PyTorch's version of argmax
            # but it returns a tuple: (max value, index of max value)
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            # In binary classification, we need to check if the
            # last later is a sigmoid (and then it produces probs)
            if isinstance(self.model, nn.Sequential) and \
                isinstance(self.model[-1], nn.Sigmoid):
                predicted = (yhat > threshold).long()
            # or something else (logits), which we need to convert
            # using a sigmoid
            else:
                predicted = (torch.sigmoid(yhat) > threshold).long()

        # How many samples got classified
        # correctly for each class
        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()
            n_correct = (predicted[y ==c] == c).sum().item()
            result.append((n_correct, n_class))
        return torch.tensor(result)

    @staticmethod
    def loader_apply(loader, func, reduce='sum'):
        results = [func(x, y) for i, (x, y) in enumerate(loader)]
        results = torch.stack(results, axis=0)

        if reduce == 'sum':
            results = results.sum(axis=0)
        elif reduce == 'mean':
            results = results.float().mean(axis=0)

        return results