import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import auc


def threshold_inout(scores_in: torch.Tensor,
                    scores_out: torch.Tensor,
                    num: int = 1000,
                    cmp=torch.gt,
                    print_auc: bool = False):
    """
    Compute OOD threshold from in- and out-of- distribution data.

    Args:
        scores_in: scores of in-distribution data with shape (N_i, ),
            where N_i is the number of in-distribution samples.
        scores_out: scores of out-of-distribution data with shape (N_o, ),
            where N_o is the number of OOD samples.
        num: number of threshold to try.
        cmp:
        print_auc:
    Return:
        Optimal threshold.
    """
    # Step 1: Define a range of possible OOD thresholds to test
    # thresholds = np.linspace(scores_in.min().item(),
    #                          scores_in.max().item(), num=num)
    if cmp == torch.gt:
        vals = scores_in.quantile(torch.tensor([0.045, 0.055],
                                               device=scores_in.device))
    else:
        vals = scores_in.quantile(torch.tensor([0.945, 0.955],
                                               device=scores_in.device))
    thresholds = np.linspace(vals[0].cpu().item(),
                             vals[1].cpu().item(), num=num)

    # Step 2: Calculate TPR and FPR for each threshold
    tpr_list = np.empty(num, dtype=np.float32)
    fpr_list = np.empty(num, dtype=np.float32)
    for i, threshold in enumerate(thresholds):
        tpr_list[i] = torch.sum(cmp(scores_in, threshold)).item()
        fpr_list[i] = torch.sum(cmp(scores_out, threshold)).item()
    tpr_list /= scores_in.shape[0]
    fpr_list /= scores_out.shape[0]

    # Step 3: Calculate the AUC of the ROC curve
    if print_auc:
        auc_score = auc(fpr_list, tpr_list)
        print(f"\tAUC score: {auc_score:.4f}")

    # Step 4: Choose the threshold
    # maximizes the Youden's index, i.e., TPR - FPR
    # youden_scores = tpr_list - fpr_list
    # opt_index = np.argmax(youden_scores[_mask])
    # Minimize false positive
    opt_index = np.argmin(fpr_list)
    opt_threshold = thresholds[opt_index]
    print(f"\tOptimal threshold: {opt_threshold:.4f}")

    return opt_threshold, fpr_list[opt_index]


class ODIN(object):
    """
    Rationale:
     1. Temperature scaling is necessary to lower the argmax prediction.
        However, too large values for the temperature give equal probabilities
        to all classes (our models are less confident than those on images).
        --> Problem: there might be classes for which the predictions are more
                     confident (majority class).
     2. The input processing is important to increase the score. However, to
        have an increase in the score I need to sum and not subtract.
        --> Different importance of features.

    """
    def __init__(self, model, temp: float = 200, device: str = 'cpu'):
        self.model = model
        self.temp = temp
        self.device = device
        #
        self._eps = None
        self._th = None
        self._q = torch.tensor([.045, .046, .047, .048, .049, .05,
                                .051, .052, .053, .054, .055])

    def perturbed_pred(self, loader, eps: float = 0.01):
        """
        Generate predictions for original data and perturbed data

        Args:
            loader: data loader.
            eps: magnitude of perturbation.
        Return:
            ...
        """
        bb = []
        after_scores, before_scores = [], []
        self.model.eval()
        for x, _ in loader:

            # Step 1: Generate the prediction for original data
            _x = torch.zeros_like(x, requires_grad=True)
            pred, _ = self.model(x + _x)
            before_scores.append((pred / self.temp).softmax(-1).cpu())

            # Step 2: Compute the perturbation (in batched form)
            loss = F.cross_entropy(pred, pred.argmax(-1), reduction='sum')
            loss.backward()
            batch = x - eps * _x.grad.detach().sign()

            # Step 3: Generate the predictions for perturbed data
            with torch.no_grad():
                _pred, z = self.model(batch)
                after_scores.append((_pred / self.temp).softmax(-1).cpu())
                bb.append(z)

        # Return both the scores
        return torch.cat(before_scores, dim=0).detach(), \
            torch.cat(after_scores, dim=0), torch.cat(bb, dim=0)

    def tune(self, in_data, ood_data):
        """
        Note that these method stores the best value for eps and delta.

        Args:
            in_data: IN-Distribution data.
            ood_data: OUT-Distribution data.
        """
        print('Tuning ODIN ...')

        # Select the best epsilon given the temperature
        best = (2, None, None)
        for eps in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]:
            print(f"  EPS={eps}:")
            # Step 1: Generate scores
            before_in, after_in, _ = self.perturbed_pred(in_data, eps)
            before_ood, after_ood, _ = self.perturbed_pred(ood_data, eps)
            scores_in = after_in.max(-1)[0]
            scores_out = after_ood.max(-1)[0]

            # Step 2:
            th, th_score = threshold_inout(scores_in, scores_out)
            if best[0] > th_score:
                print(f"\tNEW TH={th:.5f}: fpr={th_score:.5f}")
                best = (th_score, eps, th)

        # Step 3: Set values
        self._eps = best[1]
        self._th = best[2]
        return self._eps, self._th

    def predict(self, loader, delta: float = None, eps: float = None):
        """
        Predict Out-Of Distribution.

        Args:
            loader: data loader.
            delta: rejection threshold.
            eps: perturbation factor.
        Return:
             - The model predictions.
             - The OOD predictions.
        """
        _th = delta if delta is not None else self._th
        assert _th != None, "Tune ODIN before predicting!"

        # Compute score before and after the perturbation
        _eps = eps if eps is not None else self._eps
        before_scores, after_scores, _ = self.perturbed_pred(loader, _eps)
        # before_s = before_scores.max(-1)[0]
        after_s, y_pred = after_scores.max(-1)

        # NOTE: we might want to check the difference instead!!!!
        return y_pred, (after_s <= _th).int()


class Mahalanobis(object):

    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.dev = device
        self.means = []
        self.covs = []
        #
        self._eps = None
        self._th = None

    @staticmethod
    def __mahalanobis__(points: torch.Tensor, means: torch.Tensor,
                        covs: torch.Tensor):
        """ Batched Mahalanobis distance. """
        _diff = points - means
        _score = torch.bmm(_diff.unsqueeze(1), covs).squeeze(1)
        return torch.sum(_score * _diff, dim=1)

    def perturbed_pred(self, loader, eps: float = 0.01):
        """

        Args:
            loader:
            eps:
        Return:
        """
        b_scores = []
        a_scores = []
        y_preds = []
        bb = []
        self.model.eval()
        for x, _ in loader:

            # Step 1: Generate the prediction for original data
            _x = torch.zeros_like(x, requires_grad=True)
            pred, z = self.model(x + _x)

            # Step 2: Compute the perturbation (in batched form)
            y_pred = pred.argmax(-1)
            _score = self.__mahalanobis__(z, self.means[y_pred],
                                          self.covs[y_pred])
            _score.sum().backward()
            batch = x - eps * _x.grad.detach().sign()
            b_scores.append(_score.detach().cpu())

            # Step 3: Generate the predictions for perturbed data
            with torch.no_grad():
                _pred, z = self.model(batch)
                y_pred = _pred.argmax(-1)
                _score = self.__mahalanobis__(z, self.means[y_pred],
                                              self.covs[y_pred])
                a_scores.append(_score.cpu())
                y_preds.append(y_pred)
                bb.append(z)

        #
        return torch.cat(b_scores, dim=0), torch.cat(a_scores, dim=0), \
            torch.cat(y_preds, dim=0), torch.cat(bb, dim=0)

    def tune(self, in_data, ood_data,
             pnts_train: torch.Tensor, labels_train: torch.Tensor):
        print('Tuning Mahalanobis ...')

        # Step 1: Find mean and covariance for each class
        for _c in torch.unique(labels_train, sorted=True):
            _pnts = pnts_train[labels_train == _c]
            self.means.append(_pnts.mean(0))
            self.covs.append(torch.linalg.inv(torch.cov(_pnts.T)))
        self.means = torch.stack(self.means, dim=0)
        self.covs = torch.stack(self.covs, dim=0)

        # return
        # Select the best epsilon given the temperature
        best = (2, None, None)
        for eps in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            print(f"  EPS={eps}:")
            # Step 2: Generate scores
            _, scores_in, _, _ = self.perturbed_pred(in_data, eps)
            _, scores_out, _, _ = self.perturbed_pred(ood_data, eps)

            # Step 3:
            th, th_score = threshold_inout(scores_in, scores_out, cmp=torch.lt)
            if best[0] > th_score:
                print(f"\tNEW TH={th:.5}: fpr={th_score:.5f}")
                best = (th_score, eps, th)

        # Step 4: Set values
        self._eps = best[1]
        self._th = best[2]
        return self._eps, self._th

    def predict(self, loader, delta: float = None, eps: float = None):
        """
        Predict Out-Of Distribution.

        Args:
            loader: data loader.
            delta: rejection threshold.
            eps: perturbation factor.
        Return:
             - The model predictions.
             - The OOD predictions.
        """
        _th = delta if delta is not None else self._th
        assert _th != None, "Tune ODIN before predicting!"

        # Compute score before and after the perturbation
        _eps = eps if eps is not None else self._eps
        _, scores, y_pred, _ = self.perturbed_pred(loader, _eps)

        #
        return y_pred, (scores >= _th).int()
