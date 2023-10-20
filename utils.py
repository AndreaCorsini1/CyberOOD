import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics as skl_metrics


@torch.no_grad()
def predictions(model, loader, device: str = 'cpu'):
    """
    Generate predictions from a model.

    Args:
        model:
        loader:
        device:
    """
    y_true, y_pred, embed = [], [], []
    model.eval()
    # Generate the predictions
    for x, y in loader:
        pred, z = model(x.to(device))
        #
        embed.append(z)
        y_true.append(y)
        y_pred.append(pred)
    #
    return torch.cat(y_pred, dim=0), torch.cat(y_true, dim=0), \
           torch.cat(embed, dim=0)


def performance(y_true, y_pred, classes=('Benign', 'Malicious'),
                plot: bool = False):
    """

    Args:
        y_true: 1D array of true labels.
        y_pred: 1d array of predictions.
        classes: name of each class.
        plot: whether to plot the confusion matrix.
    Return:
        Confusion matrix.
    """
    cm = skl_metrics.confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(cm, columns=classes, index=classes)
    if plot:
        sns.heatmap(cm, annot=True, fmt='g',
                    linewidth=.5, annot_kws={"fontsize": 8})
        plt.yticks(rotation=30)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
    else:
        print(cm)
    # rep = skl_metrics.classification_report(y_true, y_pred,
    #                                         # output_dict=True,
    #                                         zero_division=0,
    #                                         target_names=classes)
    # print(rep)
    return cm


def fgsm(model: torch.nn.Module,
         x: torch.Tensor,
         y: torch.Tensor,
         epsilon: float = 0.1):
    """
    Construct FGSM adversarial examples on the examples x.

    Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy.
    "Explaining and harnessing adversarial examples."

    Args:
        model:
        x:
        y:
        epsilon:
    :return:
    """
    delta = torch.zeros_like(x, requires_grad=True)
    pred, _ = model(x + delta)
    if isinstance(pred, tuple):
        pred, _ = pred
    #
    loss = F.cross_entropy(pred, y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()
