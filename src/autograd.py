import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from main import x_train, y_train, x_test, y_test, w1, b1, w2, b2, new_w1, new_b1, new_w2, new_b2

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

mean = x_train.mean(dim=0)
std = x_train.std(dim=0)
epsilon = 1e-8

feature_scale = (x_train - mean) / (std + epsilon)
losses = []
x_test_scaled = (x_test - mean) / (std + epsilon)

w1 = torch.randn(30,16, requires_grad=True) 
b1 = torch.zeros(16,requires_grad=True)  
w2 = torch.randn(16,2, requires_grad=True)
b2 = torch.zeros(2,requires_grad=True)

def calculate_gradients(loss,w1,b1,w2,b2):
    loss.backward()
    dw1 = w1.grad
    db1 = b1.grad
    dw2 = w2.grad
    db2 = b2.grad
    return dw1, db1, dw2, db2

