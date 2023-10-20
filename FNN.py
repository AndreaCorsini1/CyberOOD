import torch


class Encoder(torch.nn.Module):
    """
    Dense encoder.

    Args:
        in_size: input size
        hidden_size1: size of the first layer.
        hidden_size2: size of the second layer.
        p1: dropout after first layer.
        p2: dropout after second layer.
    """
    def __init__(self,
                 in_size: int,
                 hidden_size1: int = 256,
                 hidden_size2: int = 128,
                 hidden_size3: int = 32,
                 p1: float = 0.4,
                 p2: float = 0.4,
                 p3: float = 0.4,
                 leaky_slope: float = 0.15):
        super(Encoder, self).__init__()
        #
        self.lin1 = torch.nn.Linear(in_size, hidden_size1)
        self.dropout1 = torch.nn.Dropout(p=p1)
        #
        self.lin2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = torch.nn.Dropout(p=p2)
        #
        self.lin3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.dropout3 = torch.nn.Dropout(p=p3)
        #
        self.lin4 = torch.nn.Linear(hidden_size3, 2)
        self.act = torch.nn.LeakyReLU(negative_slope=leaky_slope)

    def forward(self, x):
        """ Forward call. """
        x1 = self.act(self.lin1(x))
        x1 = self.dropout1(x1)
        #
        x2 = self.act(self.lin2(x1))
        x2 = self.dropout2(x2)
        #
        x3 = self.act(self.lin3(x2))
        x3 = self.dropout3(x3)
        #
        return self.act(self.lin4(x3))


class FNN(torch.nn.Module):
    """
    Feedforward Neural Network.

    Args:
        num_classes
    """
    name = 'FNN'

    def __init__(self, num_classes: int = 2, **kwargs):
        super(FNN, self).__init__()
        #
        self.encoder = Encoder(**kwargs)
        self.clf = torch.nn.Linear(2, num_classes)

    def forward(self, x):
        """ Forward call. """
        z = self.encoder(x)
        logits = self.clf(z)
        #
        return logits, z
