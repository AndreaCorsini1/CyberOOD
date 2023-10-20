import torch


class MCDropout(object):
    """
    Monte Carlo Dropout.

    Args:
        model: trained model containing dropout layers.
        q: quantile for determining the OOD threshold.
        p: probability of switching off neurons.
        device: either cpu or cuda
        eps: avoid division by zero
    """
    def __init__(self,
                 model: torch.nn.Module,
                 q: float = .05,
                 p: float = 0.4,
                 temp: float = 1.,
                 device: str = 'cpu',
                 eps: float = 0.000001):
        self.model = model
        self.device = device
        self.eps = eps
        self.T = temp
        self.p = p
        self.avg_th = []
        self.std_th = []
        self._q = q

    def __enable__(self):
        """ Enable dropout layer and set probability. """
        self.model.eval()
        # Enable the dropout layers in evaluation mode
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.p = self.p
                m.train()

    @torch.no_grad()
    def __predict__(self, loader, num_samples=100):
        """
        Generate prediction with dropout.

        Note that here we want to detect OOD, therefore no need to care about
        correct or wrong prediction.
        """
        self.__enable__()       # Enable dropout in the model

        mcd_preds = []
        _y = []
        for x, y in loader:
            _preds = []

            # Generate multiple prediction for the batch
            for _ in range(num_samples):
                _logits, _ = self.model(x)
                _preds.append(torch.softmax(_logits / self.T, dim=-1))
            mcd_preds.append(torch.stack(_preds, dim=1))
            _y.append(y)

        return torch.cat(mcd_preds, dim=0), torch.cat(_y, dim=0)

    def tune(self, val, num_samples: int = 100):
        """
        Extract a rejection (OOD) threshold for each class.

        Args
            val: validation loader.
            num_samples: number of sample for each input in MCD.
        Return:
            None
        """
        # Step 1: make the prediction
        mcd_preds, y_true = self.__predict__(val, num_samples)

        # Step 2: Compute thresholds
        mean = mcd_preds.mean(1)
        std = mcd_preds.std(1)
        for _c in torch.unique(y_true, sorted=True):
            _mask = y_true == _c
            self.avg_th.append(mean[_mask, _c].quantile(self._q))
            self.std_th.append(std[_mask, _c].quantile(1 - self._q))

    def uncertainty(self, batch, num_samples: int = 100):
        """
        Measure the model uncertainty in a batch.

        Args:
            batch:
            num_samples:
        :return:
        """
        self.__enable__()
        _x = batch.to(self.device)
        _preds = []

        # Generate multiple prediction for the batch
        for _ in range(num_samples):
            _p, _ = self.model(_x)
            if isinstance(_p, tuple):
                # Some models may return also the confidence
                _p = _p[0]
            _preds.append(_p.softmax(1))

        #
        mcd_preds = torch.stack(_preds, dim=1)
        mean = mcd_preds.mean(1)
        return -torch.sum(mean * torch.log(mean + self.eps), dim=1)

    def predict(self, loader, num_samples: int = 100):
        """
        OOD prediction.

        Args:
            loader: data loader.
            num_samples: number of predictions to generate for each input.
        Return:
            For each input:
                - Predicted class
                - OOD prediction
        """
        assert self.avg_th, "Tune before predict!"

        # Step1: Generate multiple predictions
        mcd_preds, _ = self.__predict__(loader, num_samples)

        # Step 2: Compute MCD scores (avg and std)
        a_scores, y_pred = mcd_preds.mean(1).max(-1)
        s_scores = mcd_preds.std(1).gather(1, y_pred.unsqueeze(1)).squeeze(1)

        # Step 3: Compute OOD predictions
        ood_pred = torch.zeros((y_pred.size(0), 2), dtype=torch.bool)
        for _c, (a_th, s_th) in enumerate(zip(self.avg_th, self.std_th)):
            _mask = y_pred == _c
            #
            ood_pred[_mask, 0] = a_scores[_mask] < a_th
            ood_pred[_mask, 1] = s_scores[_mask] > s_th

        return y_pred, ood_pred


class Confidence(object):
    """

    Args:
        model: trained model containing dropout layers.
        q: quantile for determining the OOD threshold.
        device: either cpu or cuda
    """
    def __init__(self,
                 model: torch.nn.Module,
                 q: float = .05,
                 temp: float = 2.,
                 device: str = 'cpu'):
        self.model = model
        self.T = temp
        self.device = device
        self.th = []
        self.q = torch.tensor([q], device=device)

    @torch.no_grad()
    def __predict__(self, loader, tuning: bool = False):
        """ Generate confidence. """
        self.model.eval()
        conf_preds = []
        _y = []

        for x, y in loader:
            _logits, _ = self.model(x)
            conf, y_pred = torch.softmax(_logits / self.T, dim=-1).max(-1)
            #
            _y.append(y if tuning else y_pred)
            conf_preds.append(conf)
        #
        return torch.cat(conf_preds, dim=0).squeeze(), torch.cat(_y, dim=0)

    def tune(self, val):
        """

        Args
            val: validation loader.
        Return:
        """
        # Step 1: compute temperature
        # self.model.set_temperature(val)

        # Step 2: make the prediction
        conf_preds, y_true = self.__predict__(val, tuning=True)

        #
        for _c in torch.unique(y_true, sorted=True):
            _mask = y_true == _c
            self.th.append(conf_preds[_mask].quantile(self.q))

    def predict(self, loader):
        """
        OOD prediction.

        Args:
            loader: data loader.
        """
        assert self.th is not None, "Tune before predict!"
        #
        conf_preds, y_pred = self.__predict__(loader)

        # Generate ood predictions by checking the threshold
        ood_pred = torch.zeros(y_pred.size(0), dtype=torch.bool)
        for _c, _th in enumerate(self.th):
            _mask = y_pred == _c
            ood_pred[_mask] = conf_preds[_mask] < self.th[_c]

        return y_pred, ood_pred
