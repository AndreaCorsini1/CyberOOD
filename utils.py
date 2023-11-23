import torch
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


class CenterLoss(torch.nn.Module):
    """
    Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face
    Recognition. ECCV 2016.

    Args:
        num classes: number of classes (centers).
        dim: number of embedding features. In this work, it is fixed to 2.
    """
    def __init__(self,
                 num_classes: int = 2,
                 weight: torch.Tensor = None,
                 dim: int = 2,
                 device: str = 'cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = dim
        self.device = device
        self.weight = weight
        # Initial centers
        self.centers = torch.nn.Parameter(torch.randn(num_classes, dim,
                                                      device=device))

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        """
        Forward call.

        Args:
            x: The feature matrix ().
            labels: (Tensor with shape (batch_size))
                Ground truth labels.
        """
        batch_size = x.size(0)
        # Develop square of binomial
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(
            batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
            self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        #
        classes = torch.arange(self.num_classes, dtype=torch.long,
                               device=self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        #
        if self.weight is not None:
            dist = distmat * mask.float() * self.weight
        else:
            dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


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
