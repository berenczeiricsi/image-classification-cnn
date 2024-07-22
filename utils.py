import os

import matplotlib.pyplot as plt
import torch.utils.data
from tqdm import tqdm


def set_seeds(seed: int):
    """
    A utility function for setting the seed for reproducibility

    :param seed: An arbitrary integer for the seed
    :return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)


def select_device():
    """
    Utility function for selecting the target device

    :return: Target device
    """
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device


def evaluate_model(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   device: torch.device,
                   dataset: str
                   ):
    """
     Utility function for evaluating the CNN.

    :param model: The model which should be validated
    :param dataloader: A data loader for the evaluation set
    :param loss_fn: The loss function which should be used
    :param device: A target device
    :param dataset: The dataset on which the model should be evaluated on
    :return: The evaluation loss
    """
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Evaluation on the {dataset}", position=0, leave=False):
            inputs, targets, *_ = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            eval_loss += loss_fn(outputs, targets).item()

    eval_loss /= len(dataloader)
    return eval_loss


def plot(train_losses, eval_losses, path):
    """
    Utility function for plotting the results of the training.

    :param train_losses: The losses during training
    :param eval_losses: The losses during evaluation
    :param path: Path to save the plot
    :return: None
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, eval_losses, label='Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Losses')
    plt.legend()
    plt.savefig(os.path.join(path, "plot.png"))
