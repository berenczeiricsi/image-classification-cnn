import os

import matplotlib.pyplot as plt
import torch.utils.data
from tqdm import tqdm


def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)

def select_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device

