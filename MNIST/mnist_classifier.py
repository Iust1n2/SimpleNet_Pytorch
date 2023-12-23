#!/usr/bin/env python
# coding: utf-8

# # MNIST Classifier

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import warnings
warnings.filterwarnings("ignore")


# ## Loading Data

# In[2]:


train = pd.read_csv("/home/iustin/SimpleNet_Pytorch/MNIST/data/MNIST_train.csv")
test = pd.read_csv("/home/iustin/SimpleNet_Pytorch/MNIST/data/MNIST_test.csv")
# sub = pd.read_csv("MNISTClassifier/data/sample_submission.csv")


# In[3]:


print(train.shape, test.shape)


# In[4]:


# Create features and targets
y_train = train['label']
X_train = train.drop(columns=['label'])


# ## Visualization

# In[5]:


plt.figure(figsize=(15,7))
g = sns.countplot(x=y_train, palette="CMRmap_r")
plt.title("Number of digit classes")
plt.show();


# In[6]:


fig = plt.Figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train.values[i].reshape((28, 28)), cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
plt.show();


# ## Dataset Class

# In[7]:


class DigitDataset(Dataset):
    def __init__(self, X, y, augmentations=None):
        self.inputs = (X / 255.0).to_numpy().astype(np.float32).reshape(-1,1,28,28)
        self.targets = y.values
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        if self.augmentations is not None:
            return torch.FloatTensor(self.augmentations(img=image)['image']), torch.FloatTensor(label)
        else:
            return torch.tensor(self.inputs[idx, :]), torch.tensor(self.targets[idx])


# ## DataLoaders

# In[8]:


BATCH_SIZE = 64
# Split our trainset in train and eval
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, train_size=0.8, random_state=42)


# In[9]:


# Create datasets without augmentation
train_set = DigitDataset(X_train, y_train, None)
test_set = DigitDataset(X_eval, y_eval, None)


# In[10]:


len(test_set)


# > Note:
# >
# >We can set the batch size for our test loader to the length of the dataset. This is because during testing, the focus is on assessing the model's generalization to unseen data, and there's no >need to perform batch-wise operations.

# In[11]:


# Create dataloaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=len(test_set))


# ## Building the Model

# In[12]:


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return output


# ## Training

# In[13]:


LEARNING_RATE = 0.08
MOMENTUM = 0.5
NUM_EPOCH = 10
EVAL_STEP = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use GPU if available

model = Net().to(device)
criterion = nn.CrossEntropyLoss() # Commonly used for multiclass predictions
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)


# In[14]:


from torchinfo import summary

summary(model, 
        input_size=(1, 1, 28, 28), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)


# In[15]:


def train(model, device, train_loader, criterion, optimizer, train_loss_collector):
    model.train()
    for epoch in range(NUM_EPOCH):
        epoch_loss = list()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            pred = torch.max(output.data, 1)[1]
        train_loss_collector.append(np.round(np.mean(epoch_loss),5))
        print(f'Epoch: {epoch}/{NUM_EPOCH} \t Loss: {np.round(np.mean(epoch_loss),5)}')
        if epoch % EVAL_STEP == 0:
            evaluation(model, device, test_loader, criterion, test_loss_collector)


# In[16]:


def evaluation(model, device, test_loader, criterion, test_loss_collector):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()
            test_loss_collector.append(loss.item())
    print(f'Test loss: {np.round(loss.item(), 2)} Accuracy: {correct}/{len(test_loader.dataset)}'
          f' - {(100. * correct / len(test_loader.dataset)):.2f}%')
    all_preds.extend(preds)
    all_targets.extend(target)


# In[17]:


train_loss_collector = list()
test_loss_collector = list()
all_preds = list()
all_targets = list()
print('Start training...')
train(model, device, train_loader, criterion, optimizer, train_loss_collector)
print('Finish training!')


# In[18]:


preds = [tensor.item() for tensor in all_preds]
targets = [tensor.item() for tensor in all_targets]
confusion_mtx = confusion_matrix(targets, preds)
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[19]:


plt.figure(figsize=(12,5))
sns.lineplot(x=range(1, NUM_EPOCH + 1), y=train_loss_collector, label='Train', marker='o')
sns.lineplot(x=range(1, NUM_EPOCH + 1), y=test_loss_collector, label='Test', marker='o')
plt.title('Loss for train and test runs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show();


# ## Faza 2
# 
# [GitHub Repo](https://github.com/Coderx7/SimpleNet_Pytorch/blob/master/cifar/models/simplenet.py) for simplenet310k - cel folosit pt MNIST
# 
# In procesul de evaluare, in setul de test, de salvat imaginile care au fost prezise incorect de catre model, cu tuple: `idx`, `clasa prezisa`, `clasa adevarata`. 
# De salvat imaginile intr-un director separat format de genul: `"im14_0_9"`.
# 
# > ! Split-ul in train,val,test este standardizat?
# 
# 

# In[44]:


# import models.simplenet.py

import sys

# Add the directory to the Python path in this notebook 
sys.path.append('/home/iustin/SimpleNet_Pytorch/MNIST')

from models.simplenet import simplenet_cifar_310k

# import simplenet.py import simplenet_cifar_310k
# model2 = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenet_cifar_310k", pretrained=True)


# In[28]:


import os 
os.getcwd()


# In[ ]:

import models
# from torch.utils.tensorboard import SummaryWriter

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

print('models : ',model_names)

net = models.__dict__['simplenet_cifar_310k'](num_classes=10)

# In[ ]:

def train(net, device, train_loader, criterion, optimizer, train_loss_collector):
    net.train() 
    for epoch in range(NUM_EPOCH):
        correct = 0
        epoch_loss = list()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            pred = torch.max(output.data, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss_collector.append(np.round(np.mean(epoch_loss),5))
        print(f'Epoch: {epoch}/{NUM_EPOCH} \t Loss: {np.round(np.mean(epoch_loss),5)}')
        print(f'Train Accuracy: {(100. * correct / len(train_loader.dataset)):.2f}%')
        # if epoch % EVAL_STEP == 0:
        #     evaluation(net, device, test_loader, criterion, test_loss_collector)


def evaluation(net, device, test_loader, criterion, test_loss_collector):
    net.eval() 
    # test_loss = 0
    correct_test = 0
    test_loss = list()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)
            test_loss.append(loss.item())
            preds = output.argmax(dim=1, keepdim=True)
            correct_test += preds.eq(target.view_as(preds)).sum().item()
            test_loss_collector.append(loss.item())
        print(f'Test loss: {np.round(test_loss, 2)} Accuracy: {correct_test}/{len(test_loader.dataset)}'
              f' - {(100. * correct_test / len(test_loader.dataset)):.2f}%')
    all_preds.extend(preds)
    all_targets.extend(target)

criterion = nn.CrossEntropyLoss() # Commonly used for multiclass predictions
optimizer = optim.SGD(model2.parameters(), lr=LEARNING_RATE)


# In[ ]:

train_loss_collector = list()
test_loss_collector = list()
all_preds = list()
all_targets = list()
print('Start training...')
train(net, device, train_loader, criterion, optimizer, train_loss_collector)
print('Finish training!')

# In[ ]:
abcd 

