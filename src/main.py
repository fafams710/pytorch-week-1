import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

torch.manual_seed(42)



def load_breast_cancer_():


    cancer =  load_breast_cancer()

    x = cancer.data
    y = cancer.target


    return x, y

x, y = load_breast_cancer_()



x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42 )

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_test_split (x_train,y_train,test_size=0.25,random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.25,random_state=42)

print ("y_train:", y_train.float().mean().item())

print(-torch.log(torch.tensor(1e-10)))
print(-torch.log(torch.tensor(1e-30)))
print(-torch.log(torch.tensor(0.5)))

print ("Max absolute value in x_train: ", x_train.abs().max().item())

w1 = torch.randn(30,16, ) 
b1 = torch.zeros(16, )   
z1 = torch.matmul(x_train, w1) + b1
a1 = torch.relu(z1)
b2 = torch.zeros(2)
w2 = torch.randn(16,2)
z2 = torch.matmul(a1, w2) + b2

print("gradient Z = ", z1.grad_fn)
print("z1 max: ", z1.abs().max().item())
print("z2 max: ", z2.abs().max().item())

predictions = torch.argmax(z2, dim=1)

predictions == y_train


accuracy = torch.sum(predictions == y_train).item() / len(y_train)

print(f"Accuracy: {accuracy:.4f}")

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(z2, y_train) 

print(f"Loss: {loss.item():.4f}")


dz2 = (torch.softmax(z2, dim=1) - torch.nn.functional.one_hot(y_train, num_classes=2) ) / y_train.shape[0]
da1 = torch.matmul(dz2, w2.T)
relu = torch.where(z1 > 0,1,0)
dz1 = da1 * relu
dw1 = torch.matmul(x_train.T, dz1)
db1 = dz1.sum(dim=0)
dw2 = torch.matmul(a1.T, dz2)
db2 = dz2.sum(dim=0)

print("dw1 max: ", dw1.abs().max().item())

learning_rate = 0.01

new_w1 = w1.clone()
new_b1 = b1.clone()
new_w2 = w2.clone()
new_b2 = b2.clone()


mean = x_train.mean(dim=0)
std = x_train.std(dim=0)
epsilon = 1e-8

feature_scale = (x_train - mean) / (std + epsilon)
losses = []
x_test_scaled = (x_test - mean) / (std + epsilon)

for epoch in range(200):
    z1 = torch.matmul(feature_scale, new_w1) + new_b1
    a1 = torch.relu(z1)
    z2 = torch.matmul(a1, new_w2) + new_b2



    loss = loss_fn(z2, y_train) 
    losses.append(loss.item())

    dz2 = (torch.softmax(z2, dim=1) - torch.nn.functional.one_hot(y_train, num_classes=2) ) / y_train.shape[0]
    da1 = torch.matmul(dz2, new_w2.T)
    relu = torch.where(z1 > 0,1,0)
    dz1 = da1 * relu
    dw1 = torch.matmul(feature_scale.T, dz1)
    db1 = dz1.sum(dim=0)
    dw2 = torch.matmul(a1.T, dz2)
    db2 = dz2.sum(dim=0)

    new_w1 = new_w1 - learning_rate * dw1
    new_b1 = new_b1 - learning_rate * db1
    new_w2 = new_w2 - learning_rate * dw2
    new_b2 = new_b2 - learning_rate * db2

    
    if epoch < 10 or epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")



z1_test = torch.matmul(x_test_scaled, new_w1) + new_b1
a1_test = torch.relu(z1_test)
z2_test = torch.matmul(a1_test, new_w2) + new_b2

min_loss = min(losses)
best_epoch = losses.index(min_loss)

if epoch == best_epoch:
        print(f"Best epoch: {best_epoch}, Loss: {loss.item():.4f}")

predictions_test = torch.argmax(z2_test, dim=1)

final_accuracy = torch.mean(torch.tensor(predictions_test == y_test, dtype=torch.float32)).item() * 100

print("final accuracy on test set manually calculated: ", final_accuracy , "%")
print("final loss on test set manually calculated: ", loss_fn(z2_test, y_test).item())
print("final loss on train set manually calculated: ", loss_fn(z2, y_train).item())

plt.plot(losses)
plt.yscale('log')
plt.show()
