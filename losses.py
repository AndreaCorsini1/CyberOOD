import torch


class CrossEntropy(torch.nn.Module):
    """
    Wrapper for compatibility.

    Args:
        weight: Weighting factor for each class. Shape (num classes, ).
        ignore_index: Label to ignore
    """
    def __init__(self,
                 weight: torch.Tensor = None,
                 ignore_index: int = -1):
        super(CrossEntropy, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=weight,
                                              ignore_index=ignore_index)

    def forward(self, input, target):
        """ Forward """
        return self.loss(input, target)


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
