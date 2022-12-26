import torch
import torchvision 
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from google.colab import drive
drive.mount('/content/gdrive/', force_remount=True)

#ROOT = '.data'
ROOT = "/content/gdrive/My Drive/data"

train_data = datasets.MNIST(root=ROOT,
                            train=True,
                            download=True,
                            transform = transforms.ToTensor())

test_data = datasets.MNIST(root=ROOT,
                            train=False,
                            download=True,
                            transform = transforms.ToTensor())

plt.imshow(train_data.data[0], cmap = 'gray')
plt.title(train_data.targets[0])
plt.show

train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size = 64,
                                           shuffle = False)

test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size = 64,
                                          shuffle = False)

class MNIST_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 =  nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16,16,3),
            nn.ReLU(),
        )
        #self.conv4 = nn.Sequential(
        #    nn.Conv2d(1,1,5*5*16),
        #    nn.ReLU())
        #self.conv5 = nn.Sequential(
        #    nn.Conv2d(1,1,10),
        #   nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(3*3*16, 128),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(128, 10)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
        #return F.log_softmax(x)

model = MNIST_net().to(device)

n_epoch = 10
learning_rate = 0.01

log_interval = 10

loss_epochs = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []

loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
#optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.5)

def train(dataloader, epoch):
    #model.train()
    loss_epoch = 0
    for i, (features, targets) in enumerate(dataloader):
        # Inside this loop the gradient is zeroed, label predictions are calculated, and then
        # losses are backpropagated through the network and weights are adjusted
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        # Cumulative loss is stored
        loss_epoch += loss.item()
        
        if i % log_interval == 0:
            print(f'Train Epoch: {epoch + 1} [{i * 64}/{len(dataloader.dataset)} ({100.0 * i / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
        #if i % log_interval == 0:
            # At a certain interval, the current accuracy and loss in a given epoch are printed
         #   accuracy = float(test(test_loader))
         #  print(f'Train Epoch: {epoch + 1} [{i * 64}/{len(dataloader.dataset)} ({100.0 * i / len(dataloader):.2f}%)]\tLoss: {loss.item():.2f}, Accuracy: {accuracy:.2f}')
    return loss_epoch

def test(dataloader):
    # Ensure cumulative epoch loss and correct count variables are zeroed at the begining of every epoch
    test_loss = 0
    correct = 0
    for i, (features, targets) in enumerate(dataloader):
        features, targets = features.to(device), targets.to(device)
        outputs = model(features)
        _, pred = torch.max(outputs, 1)
        test_loss += targets.size(0)
        correct += torch.sum(pred == targets)
    return 100.0 * correct / test_loss

for epoch in range(n_epoch):
    loss_epoch = 0
    loss_epoch = train(train_loader, epoch)
    loss_epochs.append(loss_epoch)
    
    train_accuracy.append(test(train_loader))
    valid_accuracy.append(test(test_loader))
    print(f"Epoch {epoch + 1}: loss: {loss_epochs[-1]:.4f}, train accuracy: {train_accuracy[-1]:.4f}, valid accuracy:{valid_accuracy[-1]:.4f}")

# Display accuracy results in a figure
plt.rcParams['figure.dpi'] = 150
plt.style.use('ggplot')

epochs = range(n_epoch)
plt.plot(epochs, [acc.cpu().detach().numpy() for acc in train_accuracy], label='Training accuracy')
plt.plot(epochs, [acc.cpu().detach().numpy() for acc in valid_accuracy], label='Test accuracy')

plt.title('Train vs Test Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

