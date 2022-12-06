import enum
import logging
import os
import sys
from pathlib import Path

import torch
from PIL.Image import Image
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from torchvision import transforms

LOG_DIR = Path("../../LOG_DIR")
NEW_LOG_DIR = None

to_tensor = transforms.ToTensor()


class ProjType(enum.IntEnum):
    None_ = 0
    Frobenius = 1
    Spectral = 2


proj_type_str = {
    ProjType.None_: "None",
    ProjType.Frobenius: "fro",
    ProjType.Spectral: "spec"
}


def binarize(input_, thresh=0.5):
    return torch.where(input_ < thresh, torch.zeros_like(input_), torch.ones_like(input_))


def accuracy(y_pred, y_true):
    with torch.no_grad():
        return balanced_accuracy_score(y_true.flatten().cpu().numpy(), binarize(y_pred).cpu().flatten().numpy())


def plot_prediction(x_path, y_path, model):
    fig, ax = plt.subplots(1, 3)
    plt.setp(ax, xticks=[], yticks=[])
    ax[0].set_title("Input")
    ax[2].set_title("Truth")
    ax[0].imshow(Image.open(x_path), cmap='Greys_r')
    ax[2].imshow(Image.open(y_path), cmap='Greys_r')
    with torch.no_grad():
        y_pred = model(to_tensor(Image.open(x_path)).cuda()).cpu()
        y_pred = binarize(y_pred)
        y = to_tensor(Image.open(y_path)).flatten(start_dim=0)
        ax[1].imshow(transforms.ToPILImage()(y_pred.view(1, 8, 8)), cmap='Greys_r')
        ax[1].set_title(f"Prediction (Acc={accuracy(y_pred, y):.3f})")
    plt.show()


def denorm(tensor):
    return (tensor + 1) / 2


def get_next_version(root_dir):
    try:
        listdir_info = os.listdir(root_dir)
    except OSError:
        print(f"Missing logger folder: {root_dir}")
        return 0

    existing_versions = []
    for d in listdir_info:
        bn = os.path.basename(d)
        if os.path.isdir(os.path.join(root_dir, d)) and bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace("/", "")
            if not dir_ver.isnumeric():
                continue
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1


def detach_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def signal_to_noise_ratio(clean_image, noisy_image):
    mean_image = clean_image.mean()
    noise = noisy_image - clean_image
    mean_noise = noise.mean()
    noise_diff = noise - mean_noise
    var_noise = (noise_diff ** 2).mean().sum()
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


def get_module_logger(name):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(get_log_path() / f"logs.log")
    fileHandler.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(fileHandler)
    return logger


def get_log_path():
    # Feels a bit hacky, but works so let's go with that for now.
    global NEW_LOG_DIR
    if NEW_LOG_DIR is None:
        NEW_LOG_DIR = LOG_DIR / f"version_{get_next_version(LOG_DIR)}"
        NEW_LOG_DIR.mkdir(exist_ok=True, parents=True)
    return NEW_LOG_DIR


class StrEnum(str, enum.Enum):
    """
    Enum where members are also (and must be) strings
    This is the exact same implementation introduced in python 3.11
    """

    def __new__(cls, *values):
        "values must already be of type `str`"
        if len(values) > 3:
            raise TypeError('too many arguments for str(): %r' % (values,))
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):
                raise TypeError('%r is not a string' % (values[0],))
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):
                raise TypeError('encoding must be a string, not %r' % (values[1],))
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):
                raise TypeError('errors must be a string, not %r' % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()
