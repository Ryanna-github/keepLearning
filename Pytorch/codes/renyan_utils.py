# some useful customized functions!
from IPython import display
import torchvision
import torchvision.transforms as transforms
import torch
from torch import nn
from torch.nn import functional as F

def user_svg_display():
    display.set_matplotlib_formats('svg')
    
def set_figsize(figsize = (3.5, 2.5)):
    user_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    
# Optimizer
# Change directly in the same memory
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    
    def forward(self, x):
        return x.view(x.shape[0], -1)

# ==================================== Fashion MNIST part ======================================
# load fashion-mnist
def load_data_fashion_mnist(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root = './data/', train = True, transform = transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root = './data/', train = False, transform = transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = 0)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False, num_workers = 0)
    return train_iter, test_iter

def load_data_fashion_mnist_resize(batch_size, resize = None, root = './data/'):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size = resize))
    trans.append(torchvision.transforms.ToTensor())
    
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root = root, train = True, download = False, transform = transform)
    mnist_test = torchvision.datasets.FashionMNIST(root = root, train = False, download = False, transform = transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False)
    
    return train_iter, test_iter


# def evaluate_accuracy_v1(data_iter, net):
#     acc_sum, n = 0.0, 0
#     for X, y in data_iter:
#         acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
#         n += y.shape[0]
#     return acc_sum/n

def evaluate_accuracy(data_iter, net, device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                # evaluation mode
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim = 1) == y.to(device)).float().sum().cpu().item()
                # training mode
                net.train()
            else:
                print("wrong net type of: {}".format(type(net)))
            n += y.shape[0]
    return acc_sum / n

# def train_fashion_mnist_v1(net, train_iter, test_iter, loss, num_epochs, batch_size, params = None, lr = None, optimizer = None):
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        
#         for X, y in train_iter:
#             y_hat = net(X)
#             l = loss(y_hat, y).sum()
            
#             if optimizer is not None:
#                 optimizer.zero_grad()
#             elif params is not None and params[0].grad is not None:
#                 for param in params:
#                     param.grad.data.zero_()
            
#             l.backward()
#             # update parameters
#             if optimizer is None:
#                 sgd(params, lr, batch_size)
#             else:
#                 optimizer.step()
            
#             train_l_sum += l.item()
#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
#             n += y.shape[0]
            
#         test_acc = evaluate_accuracy(test_iter, net)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
#               % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

import time
def train_mnist_net(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec" %
             (epoch + 1, train_l_sum/batch_count, train_acc_sum / n, test_acc, time.time() - start))
        
        
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
    
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size = x.size()[2:])