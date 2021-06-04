import custom_function
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import math
import numpy
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchsummary import summary
import matplotlib.pyplot as plt


transform = transforms.Compose(
    [transforms.Resize((28,28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    ]) 


testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=True, num_workers=0)
dataiter = iter(testloader)
images, labels = dataiter.next()


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
        results = []
        x = x.view(-1,784)
        # Without Dropout
#        results.append(x)
#        x = F.sigmoid(self.fc1(x))
#        results.append(x)
#        x = F.sigmoid(self.fc2(x))
#        results.append(x)
#        x = F.sigmoid(self.fc3(x))
#        results.append(x)
#        x = F.sigmoid(self.fc4(x))
#        results.append(x)    
        
        # With Dropout
        results.append(x)
        x = self.d1(F.sigmoid(self.fc1(x)))
        results.append(x)
        x = self.d2(F.sigmoid(self.fc2(x)))
        results.append(x)
        x = self.d3(F.sigmoid(self.fc3(x)))
        results.append(x)
        x = self.d4(F.sigmoid(self.fc4(x)))
        results.append(x)
        
        x = self.fc5(x)
        results.append(x)
        return x,results
    
    
# load model weights
net = Net()
PATH = './expt_wd_fmnist_sigmoid_net.pth'
net.load_state_dict(torch.load(PATH))
net.eval()

# compute ncut value
weights = custom_function.create_weights_mlp(net)
N, Adj = custom_function.create_adjacency_matrix(weights)
nc, clusters = custom_function.ncut(N,4,Adj)
print(nc)
custom_function.visualize_clusters(clusters)

#-----------------------------------CORRELATION COMPUTATION----------------------------------------------
activation_map = [] # store activations of each neuron for each image, index denotes image number

# store activations of whole neural network, for each image
for i in range(10000): # iterate over images
    output = net(images[i])
    val = []
    for j in range(len(output[1])):
        val.append((output[1][j].data).numpy())
    activation_map.append(val)
number_of_layers = 6
layer_sizes = [784,256,256,256,256,10]

N = 0
for i in layer_sizes:
    N += i

weights = [] # this stores zero matrices of appropriate size
for i in range(number_of_layers-1):
    weights.append(numpy.zeros((layer_sizes[i],layer_sizes[i+1])))
    
#print(activation_map[0][5][0][5]) # syntax: image no, layer output index, 0, neuron number

# store activations of each neuron 
activations_of_neurons = []
sum = 0
for i in range(number_of_layers):
    for j in range(layer_sizes[i]):
        activations = []
        for k in range(10000):
            activations.append(activation_map[k][i][0][j])
        activations_of_neurons.append(activations)
    sum += layer_sizes[i]
    
#print(len(activations_of_neurons[0]))


sums  = 0
counter = 0
counter_zeros = 0
counter_zeros_input_layer = 0
uncorr = 0
for i in range(len(weights)):
    for j in range(weights[i].shape[0]):
        current_layer_size = weights[i].shape[0]
        arr_for_j = numpy.array(activations_of_neurons[sums+j])
        if(numpy.mean(arr_for_j) < 1e-12):
            counter_zeros +=1
            if(i==0):
                counter_zeros_input_layer += 1
        for k in range(weights[i].shape[1]):
            arr_for_k = numpy.array(activations_of_neurons[sums+current_layer_size+k])
#            corr, _ = pearsonr(arr_for_j, arr_for_k)
            corr, p = spearmanr(arr_for_j, arr_for_k)
#            if(p > 0.5):
#                uncorr += 1
            if(math.isnan(corr)):
#                weights[i][j,k] = 0
                weights[i][j,k] = 0.0000000001
                counter = counter + 1
            else:
                weights[i][j,k] = abs(corr)
    sums += weights[i].shape[0]

# Test values
#print(counter)
print(counter_zeros)
print(counter_zeros_input_layer)
#print(weights[4])
#print(uncorr)
    
# Compute ncut via Spearman method    
N, Adj  = custom_function.create_adjacency_matrix(weights)
nc, clusters = custom_function.ncut(N, 4, Adj)
print(nc)
custom_function.visualize_clusters(clusters)
