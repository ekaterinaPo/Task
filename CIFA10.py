import torch
import torchvision 
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

from google.colab import drive
drive.mount('/content/gdrive/', force_remount=True)

#ROOT = '.data'
ROOT = "/content/gdrive/My Drive/data"

train_data = torchvision.datasets.CIFAR10(root=ROOT,
                            train=True,
                            download=True,
                            transform = transform)

test_data = torchvision.datasets.CIFAR10(root=ROOT,
                            train=False,
                            download=True,
                            transform = transform)



transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size = 32,
                                           shuffle = False)

test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size = 32,
                                          shuffle = False)

from torchvision.utils import make_grid

for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break

class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(3,16, 3, padding=1),                              
            nn.ReLU(),
            nn.Conv2d(16,32, 3, padding=1),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(32, 64, 3, padding=1),     
            nn.ReLU(),
            nn.Conv2d(64,128, 3, padding=1),                              
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(128, 256, 3, padding=1),     
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),                
        )
        
        self.fc1   = nn.Sequential(
            nn.Linear(4*4*256, 512),
            nn.ReLU(),
        )

        self.fc2   = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc3   = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0.2)
       
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #x = self.dropout(x)
        x = self.conv3(x)
        #x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

cn_net = CNNet().to(device)

n_epoch = 10
learning_rate = 0.01

log_interval = 10
momentum = 0.9

loss_epochs = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []

loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)

optimizer = optim.SGD(cn_net.parameters(), lr = learning_rate, momentum=momentum)

def train(dataloader, epoch):
    loss_epoch = 0
    for i, (features, targets) in enumerate(dataloader):
        # Inside this loop the gradient is zeroed, label predictions are calculated, and then
        # losses are backpropagated through the network and weights are adjusted
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = cn_net(features)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        # Cumulative loss is stored
        loss_epoch += loss.item()
        #acc_epoch += acc.item
        if i % log_interval == 0:
            # At a certain interval, the current accuracy and loss in a given epoch are printed
            #accuracy = float(test(test_loader))
            print(f'Train Epoch: {epoch + 1} [{i * 32}/{len(dataloader.dataset)} ({100.0 * i / len(dataloader):.2f}%)]\tLoss: {loss.item():.2f}')
    return loss_epoch 

def test(dataloader):
    # Ensure cumulative epoch loss and correct count variables are zeroed at the begining of every epoch
    test_loss = 0
    correct = 0
    for i, (features, targets) in enumerate(dataloader):
        features, targets = features.to(device), targets.to(device)
        outputs = cn_net(features)
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
