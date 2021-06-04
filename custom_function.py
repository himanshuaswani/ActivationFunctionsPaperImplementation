import numpy
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# function for ncut
def ncut(N, k, Adj):
    #computation of Degree Matrix D
    Deg = numpy.zeros((N,N))

    for i in range(N):
        for j in range(N):
            Deg[i,i] += Adj[i,j] # only diagonal elements non-zero
    #computation of Laplacian L
    L = Deg - Adj
    L_norm = numpy.matmul(numpy.linalg.inv(Deg),L)
    #count = 0
    #eigenvalue decomposition
    values, vectors = numpy.linalg.eig(L_norm)
    values = values.real

    #find smallest k eigenvalues indices
    order = numpy.argsort(values)
    
    # form U
    U = numpy.zeros((k,N))
    
    for i in range(k):
        for j in range(N):
            U[i,j] = vectors[j,order[i]]
        
    # K means here we go
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=1000).fit(U.transpose())

    #print kmeans clusters
    #print(kmeans.labels_)
    labels = kmeans.labels_
    
    #form clusters Xi
    C = []
    for i in range(k):
        C.append([])
        
    for i in range(N):
        for j in range(k):
            if(labels[i] == j):
                C[j].append(i)
#    sizes = []    
    #calculate n-cut
    ncut = 0
#    contribution = []
    for i in range(k):
        X = C[i]
#        sizes.append(len(X))
        Xbar = []
        W = 0
        V = 0
        for j in range(N):
            try:
                X.index(j)
            except ValueError:
                Xbar.append(j)
        for e1 in X:
            for e2 in Xbar:
                W += Adj[e1,e2]
            V += Deg[e1,e1]
        ncut += W/V
#        contribution.append(W/V)
#    print(contribution)
#    print(sizes)
#    print(ncut)
    return ncut, C

def create_adjacency_matrix(weights):
    number_of_layers = len(weights)
#    print(number_of_layers)
#    for m in weights:
#        print(m.shape)
        
    layer_sizes = [] # store different layer sizes in terms of number of neurons
    for m in weights:
        layer_sizes.append(m.shape[0])
    layer_sizes.append((weights[number_of_layers-1]).shape[1])
#    print(layer_sizes)
    
    N = 0 # denotes number of neurons in the whole network including input and output neurons
    for i in layer_sizes:
        N += i
#    print(N)
    
    cumulative = []
    sum = 0 
    for l in layer_sizes:
        sum = sum + l
        cumulative.append(sum)
    #print(cumulative)
    Adj = numpy.zeros((N,N)) # our adjacency matrix
    count = 0
    sum = 0
    for i in range(number_of_layers):
        current_layer_size = weights[i].shape[0]
        next_layer_size = weights[i].shape[1]
        for m in range(current_layer_size):
            for n in range(next_layer_size):
                Adj[sum + m,sum + current_layer_size+n] = abs((weights[i][m,n]))
                Adj[sum + current_layer_size+n,sum + m] = Adj[sum + m,sum + current_layer_size+n]
                count = count + 1
        sum += current_layer_size
    return N,Adj

def create_weights_mlp(net):
    weights = [] # store weight matrices between layers
    for param in net.parameters():
        M = ((param.data).numpy()).transpose()
        s = M.shape
        if(len(s) == 2):
            W = numpy.zeros(s)
            for i in range(s[0]):
                for j in range(s[1]):
                    W[i,j] = M[i,j]
            weights.append(W)
    return weights

def visualize_clusters(c):
    N = 1818
    regions = numpy.zeros(N)
    for i in range(len(c)):
        for j in c[i]:
            regions[j] = i

    layer_sizes = [784,256,256,256,256,10]
    sums = 0
    layer = 0

    fig, p = plt.subplots(2,2)
    for i in layer_sizes:
        for j in range(sums,sums+i):
            if(i == 784):
                if(regions[j] == 0):
                    p[0][0].plot(j,layer,'ob')
                elif(regions[j] == 1):
                    p[0][1].plot(j,layer,'or')
                elif(regions[j] == 2):
                    p[1][0].plot(j,layer,'og')
                elif(regions[j] == 3):
                    p[1][1].plot(j,layer,'oy')
            elif(i == 256):
                if(regions[j] == 0):
                    p[0][0].plot(j-sums+263,layer,'ob')
                elif(regions[j] == 1):
                    p[0][1].plot(j-sums+263,layer,'or')
                elif(regions[j] == 2):
                    p[1][0].plot(j-sums+263,layer,'og')
                elif(regions[j] == 3):
                    p[1][1].plot(j-sums+263,layer,'oy')
            elif(i == 10):
                if(regions[j] == 0):
                    p[0][0].plot(j-1420,layer,'ob')
                elif(regions[j] == 1):
                    p[0][1].plot(j-1420,layer,'or')
                elif(regions[j] == 2):
                    p[1][0].plot(j-1420,layer,'og')
                elif(regions[j] == 3):
                    p[1][1].plot(j-1420,layer,'oy')
        layer += 1
        sums += i
    plt.show()