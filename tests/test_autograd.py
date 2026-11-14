import torch
import numpy as np
import matplotlib.pyplot as plt
import src.main as model
model.forward_pass, model.load_and_split_breast_cancer_, model.data_standardization,model.w1, model.b1, model.w2, model.b2


def test_manual_gradient_to_autograd_():
    torch.manual_seed(42)

    x_train, x_test, y_train, y_test = model.load_and_split_breast_cancer_()  
    feature_scale, x_test_scaled = model.data_standardization(x_train, x_test)    
    z1, a1, z2 = model.forward_pass()

    w1 = model.w1.clone().detach().requires_grad_(True)
    b1 = model.b1.clone().detach().requires_grad_(True)
    w2 = model.w2.clone().detach().requires_grad_(True)
    b2 = model.b2.clone().detach().requires_grad_(True)
    feature_scale = torch.tensor(feature_scale,dtype=torch.float32)

    z1 = torch.matmul(feature_scale, w1) + b1
    a1 = torch.relu(z1)
    z2 = torch.matmul(a1, w2) + b2


    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    
    loss = cross_entropy_loss(z2, y_train)

    loss.backward()

    dz2 = (torch.softmax(z2, dim=1) - torch.nn.functional.one_hot(y_train, num_classes=2) ) / y_train.shape[0]
    da1 = torch.matmul(dz2, w2.T)
    relu = torch.where(z1 > 0.0,1.0,0.0)
    dz1 = da1 * relu
    dw1_manual_comp = torch.matmul(feature_scale.T, dz1)
    db1_manual_comp = dz1.sum(dim=0)
    dw2_manual_comp = torch.matmul(a1.T, dz2)
    db2_manual_comp = dz2.sum(dim=0)

    assert torch.allclose(dw1_manual_comp, w1.grad, atol=1e-6)
    assert torch.allclose(db1_manual_comp, b1.grad, atol=1e-6)
    assert torch.allclose(dw2_manual_comp, w2.grad, atol=1e-6)
    assert torch.allclose(db2_manual_comp, b2.grad, atol=1e-6)





