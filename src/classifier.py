import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import torch.nn.functional as F

torch.manual_seed(42)

class BreastCancerClassifier(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier, self).__init__()
        self.w1 = nn.Linear(30,16)
        self.w2 = nn.Linear(16,2)
        self.a1 = nn.ReLU()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)


    def forward(self,x):
        w1 = self.w1(x)
        a1 = self.a1(w1)
        w2 = self.w2(a1)
        return (w2)
    
    def fit(self, x_train, y_train,epochs=200):
        losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            predictions = self(x_train)
            loss = self.loss_function(predictions, y_train)
            losses.append(loss.item())
            loss.backward()
            min_loss = min(losses)
            best_epoch = losses.index(min_loss)
            self.optimizer.step()

            if epoch < 10 or epoch % 20 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        if epoch == best_epoch:
            print(f"Best epoch: {best_epoch}, Loss: {loss.item():.4f}")

        return losses,best_epoch


model = BreastCancerClassifier()

def load_breast_cancer_():


    cancer =  load_breast_cancer()

    x = cancer.data
    y = cancer.target

    return x, y

def load_and_split_breast_cancer_():
    x, y = load_breast_cancer_()
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42 )
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_and_split_breast_cancer_()

def data_standardization(x_train, x_test):
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0)
    epsilon = 1e-8

    feature_scale = (x_train - mean) / (std + epsilon)
    x_test_scaled = (x_test - mean) / (std + epsilon)

    return feature_scale, x_test_scaled
feature_scale, x_test_scaled = data_standardization(x_train, x_test)

model.fit(feature_scale,y_train)

with torch.no_grad():
    predictions = model(x_test_scaled)

predictions = torch.argmax(predictions, dim=1)
correct_predictions = torch.sum(predictions == y_test).item()
accuracy = correct_predictions / len(y_test)
print(f"Accuracy: {accuracy * 100 :.4f}% ")
print(f"final loss on test data: {model.loss_function(model(x_test_scaled), y_test).item():.4f}")
print(f"final loss on train data: {model.loss_function(model(feature_scale), y_train).item():.4f}")