import torch
import torch.nn as nn
from more_itertools import windowed


def init_mlp(
    layer_sizes, hidden_activation=nn.ReLU, output_activation=nn.Identity
) -> nn.Sequential:
    """
    Returns a PyTorch multilayer perceptron

    layer_sizes: Tuple of sizes of layers. Index 0 is the input layer
    hidden_activation: Function to use in the hidden layers
    output_activation: Function to use for the output layer 
    """
    layers = []
    for input_size, output_size in windowed(layer_sizes, n=2):
        layers.append(nn.Linear(input_size, output_size))
        layers.append(hidden_activation())
    layers[-1] = output_activation()
    return nn.Sequential(*layers)


def as_tensor(arr, device=None):
    if not isinstance(arr, torch.Tensor):
        arr = torch.Tensor(arr)
    if device and device != arr.device:
        arr = torch.Tensor(arr).to(device)
    return arr


def as_numpy(tensor):
    return tensor.cpu().detach().numpy()


def get_device(device=None) -> torch.device:
    return torch.device(device if (torch.cuda.is_available() and device) else "cpu")

