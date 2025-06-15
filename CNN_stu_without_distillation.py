import torch
import torch.utils
import torch.utils.data
import torchvision 
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets as datasets
from torch.optim import SGD
import torch.nn.functional as F
batch_size = 64
num_classes = 10
device = torch.device('cuda')

#Loading inbuilt dataset CIFAR10 , defining our training and testing data
all_transforms = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2023, 0.1994, 0.2010])])

#Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

train = torchvision.datasets.CIFAR10(root= './data',
                                     train= True,
                                     transform= train_transform,
                                     download= True)


test = torchvision.datasets.CIFAR10(root = './data',
                                    train = False,
                                    transform= all_transforms,
                                    download= True)

train_loader = torch.utils.data.DataLoader(dataset= train, 
                                           batch_size = batch_size,
                                           shuffle= True)
test_loader = torch.utils.data.DataLoader(dataset= test,
                                          batch_size= batch_size,
                                          shuffle= True)
class StudentCNN_nodistil(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentCNN_nodistil, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)  # adjust if needed
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
criterion = nn.CrossEntropyLoss()
student_nodistil = StudentCNN_nodistil(num_classes=10).to(device)
optimizer_2 = SGD(student_nodistil.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)
for epoch in range(64):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #Forward pass
        output = student_nodistil (images)
        loss = criterion(output, labels)

        #Back prop
        optimizer_2.zero_grad()
        loss.backward()
        optimizer_2.step()

        print(epoch, loss)


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = student_nodistil(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {:.2f} %'.format(100 * correct / total))



