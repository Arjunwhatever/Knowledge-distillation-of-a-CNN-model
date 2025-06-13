#Importing ofc
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
#Teacher
class CNN(nn.Module):
    
    def __init__(self, num_classes = 10):
        super(CNN, self).__init__()

        self.clayer_1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size= 3, padding = 1) 
        #self.clayer_2 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 3,padding = 1)
        self.max_pool1= nn.MaxPool2d(kernel_size= 2,stride = 1)

        self.clayer_3 = nn.Conv2d( in_channels= 32, out_channels= 64,kernel_size= 3,padding= 1)
        #self.clayer_4 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3, padding= 1)
        self.max_pool2 = nn.MaxPool2d(kernel_size= 2, stride = 2)
        
        self.fc1 = nn.Linear(14400, 128)
        self.relu = nn.ReLU()
        self.fc2 =nn.Linear(128,num_classes)

    def forward(self, x):
        out = self.clayer_1(x)
        #out = self.clayer_2(out)
        out = self.max_pool1(out)

        out = self.clayer_3(out)
        #out = self.clayer_4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


teacher_model = CNN(num_classes = 10)
teacher_model.to(device)


# Load and freeze teacher
teacher_model = CNN(num_classes=10).to(device)
teacher_model.load_state_dict(torch.load('cnn_model.pth'))
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False


class StudentCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentCNN, self).__init__()
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


def distillation_loss(student_logits, teacher_logits, true_labels, T=3.0, alpha=0.7):
    # Soft loss
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    # Hard label loss
    hard_loss = F.cross_entropy(student_logits, true_labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

student_model = StudentCNN(num_classes=10).to(device)
optimizer = SGD(student_model.parameters(), lr=0.01, weight_decay=0.0005, momentum=0.9)

for epoch in range(30):
    student_model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(images)

        student_logits = student_model(images)
        loss = distillation_loss(student_logits, teacher_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/30], Loss: {loss.item():.4f}")

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = student_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {:.2f} %'.format(100 * correct / total))

    print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = teacher_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {:.2f} %'.format(100 * correct / total))

    print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))
