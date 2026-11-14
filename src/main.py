import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix

torch.manual_seed(42)

if False and os.path.exists('best_model.pth'):
    checkpoint = torch.load('best_model.pth')
    epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
    w1 = checkpoint['model_state_dict']['w1']
    b1 = checkpoint['model_state_dict']['b1']
    w2 = checkpoint['model_state_dict']['w2']
    b2 = checkpoint['model_state_dict']['b2']
    best_epoch = checkpoint['epoch']
    start_epoch = epoch + 1
    print("resume training from last epoch,",start_epoch, "with best loss:", best_loss)
else:
    start_epoch = 0
    w1 = torch.randn(30,16)
    b1 = torch.zeros(16,)
    w2 = torch.randn(16,2)
    b2 = torch.zeros(2,)
    best_loss = float('inf')
    print("start training from scratch")




def load_breast_cancer_():


    cancer =  load_breast_cancer()

    x = cancer.data
    y = cancer.target

    return x, y
x, y = load_breast_cancer_()



def load_and_split_breast_cancer_():
    x, y = load_breast_cancer_()
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42 )
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = load_and_split_breast_cancer_()

print ("y_train:", y_train.float().mean().item())

def data_standardization(x_train, x_test):
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0)
    epsilon = 1e-8

    feature_scale = (x_train - mean) / (std + epsilon)
    x_test_scaled = (x_test - mean) / (std + epsilon)

    return feature_scale, x_test_scaled
feature_scale, x_test_scaled = data_standardization(x_train, x_test)


def forward_pass(): 
    z1 = torch.matmul(feature_scale, w1) + b1
    a1 = torch.relu(z1)
    z2 = torch.matmul(a1, w2) + b2

    return z1, a1, z2, 

z1, a1, z2,= forward_pass()

print("gradient Z = ", z1.grad_fn)
print("z1 max: ", z1.abs().max().item())
print("z2 max: ", z2.abs().max().item())

def prediction_accuracy():
    predictions = torch.argmax(z2, dim=1)
    accuracy = torch.sum(predictions == y_train).item() / len(y_train)
    print(f"Training Accuracy: {accuracy:.4f}")
    return accuracy
accuracy = prediction_accuracy()

def loss_function():
    _, _, z2 = forward_pass()
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(z2, y_train) 
    print(f"Loss: {loss.item():.4f}")
    return loss, loss_fn
loss,loss_fn = loss_function()

def backward_pass():
    dz2 = (torch.softmax(z2, dim=1) - torch.nn.functional.one_hot(y_train, num_classes=2) ) / y_train.shape[0]
    da1 = torch.matmul(dz2, w2.T)
    relu = torch.where(z1 > 0,1,0)
    dz1 = da1 * relu
    dw1 = torch.matmul(feature_scale.T, dz1)
    db1 = dz1.sum(dim=0)
    dw2 = torch.matmul(a1.T, dz2)
    db2 = dz2.sum(dim=0)

    return dz2, da1, relu, dz1, dw1, db1, dw2, db2

dz2, da1, relu, dz1, dw1, db1, dw2, db2 = backward_pass()

print("dw1 max: ", dw1.abs().max().item())

def parameter_updates():
    learning_rate = 0.01
    new_w1 = w1.clone()
    new_b1 = b1.clone()
    new_w2 = w2.clone()
    new_b2 = b2.clone()
    return new_w1, new_b1, new_w2, new_b2, learning_rate

new_w1, new_b1, new_w2, new_b2, learning_rate = parameter_updates()

def train_model():
    global best_loss
    losses = []
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    epochs_list = []
    global best_epoch
    x_train, x_test, y_train, y_test = load_and_split_breast_cancer_()
    feature_scale, x_test_scaled = data_standardization(x_train, x_test)
    z1, a1, z2, = forward_pass()
    loss, loss_fn = loss_function()
    new_w1, new_b1, new_w2, new_b2, learning_rate = parameter_updates()
    dz2, da1, relu, dz1, dw1, db1, dw2, db2 = backward_pass()

    for epoch in range(start_epoch, start_epoch + 200):
        learning_rate = 0.01
        z1 = torch.matmul(feature_scale, new_w1) + new_b1
        a1 = torch.relu(z1)
        z2 = torch.matmul(a1, new_w2) + new_b2
        loss = loss_fn(z2, y_train) 
        train_losses.append(loss.item())
        predictions = torch.argmax(z2, dim=1)
        train_accuracy = (predictions == y_train).float().mean().item() * 100
        train_accuracies.append(train_accuracy)
        epochs_list.append(epoch + 1)
        losses.append(loss.item())

        dz2 = (torch.softmax(z2, dim=1) - torch.nn.functional.one_hot(y_train, num_classes=2) ) / y_train.shape[0]
        da1 = torch.matmul(dz2, new_w2.T)
        relu = torch.where(z1 > 0,1,0)
        dz1 = da1 * relu
        dw1 = torch.matmul(feature_scale.T, dz1)
        db1 = dz1.sum(dim=0)
        dw2 = torch.matmul(a1.T, dz2)
        db2 = dz2.sum(dim=0)

        z1_test = torch.matmul(x_test_scaled, new_w1) + new_b1
        a1_test = torch.relu(z1_test)
        z2_test = torch.matmul(a1_test, new_w2) + new_b2

        predictions_test = torch.argmax(z2_test, dim=1)

        test_accuracy = (predictions_test == y_test).float().mean().item() * 100

        test_accuracies.append(test_accuracy)

        current_loss = loss.item()
        checkpoint = {
            'epoch': epoch,
            'loss': current_loss,
            'model_state_dict': {
                'w1': new_w1,
                'b1': new_b1,
                'w2': new_w2,
                'b2': new_b2
            }
        }

        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            torch.save(checkpoint, 'best_model.pth')

        new_w1 = new_w1 - learning_rate * dw1
        new_b1 = new_b1 - learning_rate * db1
        new_w2 = new_w2 - learning_rate * dw2
        new_b2 = new_b2 - learning_rate * db2
        
        min_loss = min(losses)


        if epoch < 10 or epoch % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    if epoch == best_epoch:
        print(f"Best epoch: {best_epoch}, Loss: {loss.item():.6f}")
        print("Checkpoint saved: Best model updated.")
            

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_list, train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(epochs_list, train_accuracies)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.subplot(1, 3, 3)
    plt.plot(epochs_list, test_accuracies)
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.show()

    return new_w1, new_b1, new_w2, new_b2, min_loss, best_epoch,checkpoint,




def evaluate_model(new_w1, new_b1, new_w2, new_b2):
    checkpoint = torch.load('best_model.pth')

    w1 = checkpoint['model_state_dict']['w1']
    b1 = checkpoint['model_state_dict']['b1']
    w2 = checkpoint['model_state_dict']['w2']
    b2 = checkpoint['model_state_dict']['b2']
    z1_test = torch.matmul(x_test_scaled, new_w1) + new_b1
    a1_test = torch.relu(z1_test)
    z2_test = torch.matmul(a1_test, new_w2) + new_b2
    z1_train = torch.matmul(feature_scale, new_w1) + new_b1
    a1_train = torch.relu(z1_train)
    z2_train = torch.matmul(a1_train, new_w2) + new_b2

    predictions_test = torch.argmax(z2_test, dim=1)
    final_accuracy = (predictions_test == y_test).float().mean().item() * 100
    print(f"final accuracy on test set manually calculated: {final_accuracy:.6f}%")
    print(f"final loss on test set manually calculated: {loss_fn(z2_test, y_test).item():.6f}")
    print(f"final loss on train set manually calculated: {loss_fn(z2_train, y_train).item():.6f}")





def main():
    new_w1, new_b1, new_w2, new_b2, _, _, _ = train_model()
    evaluate_model(new_w1, new_b1, new_w2, new_b2)

def evaluate_checkpoint(checkpoint, x_test, y_test):
    weights = checkpoint["model_state_dict"]
    w1, b1, w2, b2 = weights["w1"], weights["b1"], weights["w2"], weights["b2"]

    if "mean" in checkpoint:
        mean, std = checkpoint["mean"], checkpoint["std"]
    else:
        x_tr, _, _, _ = load_and_split_breast_cancer_()
        mean, std = x_tr.mean(dim=0), x_tr.std(dim=0)
    x_test_scaled = (x_test - mean) / (std + 1e-8)

    with torch.no_grad():
        z1 = torch.matmul(x_test_scaled, w1) + b1
        a1 = torch.relu(z1)
        z2 = torch.matmul(a1, w2) + b2

    predictions = torch.argmax(z2, dim=1)
    return (predictions == y_test).float().mean().item()

if __name__ == "__main__":

    import argparse, json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Breast Cancer Neural Network Classifier"
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--split", action="store_true")
    group.add_argument("--train", action="store_true")
    group.add_argument("--evaluate", action="store_true")

    parser.add_argument("--smoke", action="store_true")

    parser.add_argument("--out", type=str, default="data/split_indices.json")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--metrics-out", type=str, default="metrics.json")

    args = parser.parse_args()

    x_train, x_test, y_train, y_test = load_and_split_breast_cancer_()

    if args.split:
        _, y_all = load_breast_cancer_()
        indices = np.arange(len(y_all))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "train": train_idx.tolist(),
            "test": test_idx.tolist(),
        }, indent=2))

    elif args.train:
        learning_rate = 0.01
        epochs = 200
        if args.smoke:
            epochs = 10
        else:
            epochs = 200

        new_w1, new_b1, new_w2, new_b2, min_loss, best_epoch, checkpoint = train_model()
        

        out_path = Path(args.checkpoint)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, out_path)

    elif args.evaluate:

        saved_checkpoint = torch.load(
            args.checkpoint,
            map_location=torch.device("cpu"),
            weights_only=True
        )

        acc = evaluate_checkpoint(saved_checkpoint, x_test, y_test)

        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(json.dumps({
            "test_accuracy": acc
        }, indent=2))

    else:
        x_train, x_test, y_train, y_test = load_and_split_breast_cancer_()
        new_w1, new_b1, new_w2, new_b2, min_loss, best_epoch, checkpoint, learning_rate = train_model(x_train, y_train, learning_rate=0.01, epochs=200, verbose=True)
        test_acc = evaluate_model(checkpoint, x_test, y_test)

        print(f"Final test accuracy: {test_acc:.4f}")
        print(f"Final test loss: {loss_fn(torch.matmul(torch.relu(torch.matmul(x_test_scaled, new_w1) + new_b1), new_w2) + new_b2, y_test).item():.4f}")
        print(f"Final train loss: {loss_fn(torch.matmul(torch.relu(torch.matmul(feature_scale, new_w1) + new_b1), new_w2) + new_b2, y_train).item():.4f}")
        print(f"Best epoch: {best_epoch}")

#this is the for the checking that will be used as a pull request