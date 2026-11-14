import torch
from src.main import load_and_split_breast_cancer_, data_standardization, forward_pass, load_breast_cancer_ 
from src.classifier import BreastCancerClassifier
import pytest



def test_shapes():
    x_train, x_test, y_train, y_test = load_and_split_breast_cancer_()
    feature_scale, x_test_scaled = data_standardization(x_train, x_test)
    z1, a1, z2 = forward_pass()
    assert x_train.shape[1] == 30
    assert x_test.shape[1] == 30
    assert y_train.shape[0] == x_train.shape[0]
    assert y_train.ndim == 1
    assert y_test.ndim == 1
    assert y_test.shape[0] == x_test.shape[0]
    assert z1.shape == (x_train.shape[0], 16), f"Expected z1 shape {(x_train.shape[0], 16)}, but got {z1.shape}"

def test_dtypes():
    x_train, x_test, y_train, y_test = load_and_split_breast_cancer_()

    assert x_train.dtype == torch.float32
    assert x_test.dtype == torch.float32
    assert y_train.dtype == torch.long
    assert y_test.dtype == torch.long

def test_forward_pass():
    model = BreastCancerClassifier()
    x = torch.randn(10, 30)

    output = model(x)

    assert output.shape == (10,2)

    if x.shape [1] != 30:
        raise ValueError(f"Expected input 30 dimensions, but got {x.shape[1]} dimensions.")
    if x.shape[0] <= 0:
        raise ValueError(f"Expected input with positive number but got {x.shape[0]} samples.")

def test_weights_check():
    torch.manual_seed(42)
    model = BreastCancerClassifier()
    w1 = torch.randn(30,16)
    b1 = torch.zeros(16,)
    w2 = torch.randn(16,2)
    b2 = torch.zeros(2,)
    with torch.no_grad():
        model.w1.weight.copy_(w1.T)
        model.w2.weight.copy_(w2.T)
        model.w1.bias.copy_(b1)
        model.w2.bias.copy_(b2)
    assert torch.equal(w1.T, model.w1.weight)
    assert torch.equal(b1, model.w1.bias)
    assert torch.equal(w2.T, model.w2.weight)
    assert torch.equal(b2, model.w2.bias)


def test_final_loss():
    torch.manual_seed(42)
    x_train, x_test, y_train, y_test = load_and_split_breast_cancer_()
    feature_scale, x_test_scaled = data_standardization(x_train, x_test)
    w1 = torch.randn(30,16)
    w2 = torch.randn(16,2)
    b1 = torch.zeros(16,)
    b2 = torch.zeros(2,)
    x_train, x_test, y_train, y_test = load_and_split_breast_cancer_()
    z1, a1, z2 = forward_pass()
    model = BreastCancerClassifier()
    with torch.no_grad():
        model.w1.weight.copy_(w1.T)
        model.w2.weight.copy_(w2.T)
        model.w1.bias.copy_(b1)
        model.w2.bias.copy_(b2)
    loss_fn = torch.nn.CrossEntropyLoss()
    predictions = model(feature_scale)
    loss = loss_fn(z2, y_train)
    expected_loss = loss_fn(predictions, y_train)
    assert torch.isclose(loss, expected_loss, atol=1e-6)

def test_checkpoint_round_trip():
    torch.manual_seed(42)
    model = BreastCancerClassifier()
    checkpoint = model.state_dict()
    torch.save(checkpoint, 'temp_checkpoint.pth')
    new_model = BreastCancerClassifier()
    new_model.load_state_dict(torch.load('temp_checkpoint.pth'))
    x = torch.randn(16,30)
    original_output = model(x)
    new_output = new_model(x)
    assert torch.allclose(original_output, new_output, atol=1e-6)