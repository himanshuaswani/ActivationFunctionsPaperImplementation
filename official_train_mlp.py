import numpy
import torch
import torchvision
import torchvision.transforms as transforms
import custom_function
import matplotlib.pyplot as plt
transform = transforms.Compose(
    [transforms.ToTensor()])

transform = transforms.Compose(
    [transforms.Resize((28,28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ]) 

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=0)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    numpyimg = img.numpy()
    plt.imshow(numpy.transpose(numpyimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()


# show images
imshow(torchvision.utils.make_grid(images))

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,256)
        self.d1 = nn.Dropout()
        self.fc2 = nn.Linear(256,256)
        self.d2 = nn.Dropout()
        self.fc3 = nn.Linear(256,256)
        self.d3 = nn.Dropout()
        self.fc4 = nn.Linear(256,256)
        self.d4 = nn.Dropout()
        self.fc5 = nn.Linear(256,10)
        

    def forward(self, x):
        x = x.view(-1,784)
        # Without Dropout
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = F.relu(self.fc3(x))
#        x = F.relu(self.fc4(x))
        
        # With Dropout
        x = self.d1(F.relu(self.fc1(x)))
        x = self.d2(F.relu(self.fc2(x)))
        x = self.d3(F.relu(self.fc3(x)))
        x = self.d4(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x

# load model weights
net = Net()
PATH = './expt_wd_mnist_relu_net.pth'

# define loss and optimizer functions
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inumpyuts; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 250 == 249:             # print every fixed number mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 250))
    if epoch % 5 == 0:
        # save model
        torch.save(net.state_dict(), PATH)

net.eval()
# calculate test loss
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))        

# compute ncut
weights = custom_function.create_weights_mlp(net)
N, Adj = custom_function.create_adjacency_matrix(weights)
nc, clusters = custom_function.ncut(N,4,Adj)
print(nc)
custom_function.visualize_clusters(clusters)