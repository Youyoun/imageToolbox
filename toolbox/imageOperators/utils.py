import enum

import torch
from matplotlib import pyplot as plt
from PIL.Image import Image
from sklearn.metrics import balanced_accuracy_score
from torchvision import transforms

to_tensor = transforms.ToTensor()


class ProjType(enum.IntEnum):
    None_ = 0
    Frobenius = 1
    Spectral = 2


proj_type_str = {ProjType.None_: "None", ProjType.Frobenius: "fro", ProjType.Spectral: "spec"}


def binarize(input_, thresh=0.5):
    return torch.where(input_ < thresh, torch.zeros_like(input_), torch.ones_like(input_))


def accuracy(y_pred, y_true):
    with torch.no_grad():
        return balanced_accuracy_score(
            y_true.flatten().cpu().numpy(), binarize(y_pred).cpu().flatten().numpy()
        )


def plot_prediction(x_path, y_path, model):
    fig, ax = plt.subplots(1, 3)
    plt.setp(ax, xticks=[], yticks=[])
    ax[0].set_title("Input")
    ax[2].set_title("Truth")
    ax[0].imshow(Image.open(x_path), cmap="Greys_r")
    ax[2].imshow(Image.open(y_path), cmap="Greys_r")
    with torch.no_grad():
        y_pred = model(to_tensor(Image.open(x_path)).cuda()).cpu()
        y_pred = binarize(y_pred)
        y = to_tensor(Image.open(y_path)).flatten(start_dim=0)
        ax[1].imshow(transforms.ToPILImage()(y_pred.view(1, 8, 8)), cmap="Greys_r")
        ax[1].set_title(f"Prediction (Acc={accuracy(y_pred, y):.3f})")
    plt.show()


def denorm(tensor):
    return (tensor + 1) / 2


def detach_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def signal_to_noise_ratio(clean_image, noisy_image):
    mean_image = clean_image.mean()
    noise = noisy_image - clean_image
    mean_noise = noise.mean()
    noise_diff = noise - mean_noise
    var_noise = (noise_diff**2).mean().sum()
    if var_noise == 0:
        snr = 100
    else:
        snr = (torch.log10(mean_image / var_noise)) * 20
    return snr.item()


def isiterable(element):
    try:
        iter(element)
        return True
    except TypeError:
        return False
