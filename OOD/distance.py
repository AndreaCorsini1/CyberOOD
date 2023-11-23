import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils import predictions


class KNNDetector(object):
    """
    K-Nearest Neighbors Detector.
    """
    def __init__(self,
                 model,
                 k: int = 15,
                 leaf_size: int = 50,
                 q: float = 0.95,
                 device: str = 'cpu'):
        self.k = k
        self.leaf_size = leaf_size
        self._q = q
        self.device = device
        #
        self.model = model
        self.knns = []
        self.ood_th = []

    @staticmethod
    def _batching(knn, queries, batch_size: int = 1024):
        _size = queries.shape[0]
        _dists = [
            knn.kneighbors(queries[i:i + batch_size], return_distance=True)[0]
            for i in range(0, _size, batch_size)
        ]
        return np.concatenate(_dists, axis=0)

    def tune(self, dl_train, dl_val, alpha: float = 1.):
        """
        Compute the confidence intervals for in-distribution data.
        Pnts should be the embeddings of train or validation data.

        Args:
            dl_train:
            dl_val:
            alpha:
        Return:
            None
        """
        #
        _yp_train, y_train, pnts_train = predictions(self.model, dl_train,
                                                     device=self.device)
        yp_train = _yp_train.argmax(-1)
        _yp_val, y_val, pnts_val = predictions(self.model, dl_val,
                                               device=self.device)
        yp_val = _yp_val.argmax(-1)

        # Find interval for in-distribution data
        for _c in range(_yp_train.size(-1)):
            _mask = torch.logical_and(yp_train == _c, yp_train == y_train)
            _len = torch.sum(_mask).item()
            assert _len > 0, f"No train sample for class {_c}"
            _knn = NearestNeighbors(
                n_neighbors=self.k,
                leaf_size=self.leaf_size
            ).fit(pnts_train[_mask].numpy())
            self.knns.append(_knn)

            # Sample num_queries points to reduce computational cost
            _mask = torch.logical_and(yp_val == _c, yp_val == y_val)
            _len = torch.sum(_mask).item()
            assert _len > 0, f"No val sample for class {_c}"
            _idx = _mask.float().multinomial(int(_len * alpha))
            _max = self._batching(_knn, pnts_val[_idx].numpy()).max(-1)
            self.ood_th.append(np.quantile(_max, self._q))

    def from_predictions(self, pnts: torch.Tensor, y_pred: torch.Tensor):
        """
        Compute OOD-prediction from model embeddings and predictions.

        Args:
            pnts: embeddings of input samples
            y_pred: predictions generated byt the model
        Return:
        """
        #
        size = pnts.size(0)
        ood_pred = torch.zeros((size, 1), dtype=torch.bool)
        for _c in torch.unique(y_pred, sorted=True):
            _mask = y_pred == _c

            #
            _knn = self.knns[_c.item()]
            _dists = self._batching(_knn, pnts[_mask].numpy())

            _max = _dists.max(-1)
            ood_pred[_mask, 0] = torch.from_numpy(_max > self.ood_th[_c.item()])

        return ood_pred.int()

    def predict(self, loader):
        """
        Generate model and OOD predictions.

        Args:
            loader: data loader.
        Return:
             - The model predictions.
             - The OOD predictions, true if OOD.
        """
        #
        preds, y_truth, z = predictions(self.model, loader, device=self.device)
        y_preds = preds.argmax(-1)
        #
        return y_preds, self.from_predictions(z, y_preds)


class SilhoDetector(object):
    """
    Silhouette-based Detector.
    """
    def __init__(self, model,
                 q: float = 0.01,
                 device: str = 'cpu'):
        self.model = model
        self.centers = None
        self.device = device
        self._q = q

        #
        self.samples = None
        self.ood_th = []

    def __similarity__(self, pnts):
        """
        Use Simplified Silhouette (a.k.a. Medoid Silhouette).

        Other sims:
            c_sim = F.cosine_similarity(pnts.unsqueeze(1),
                                        self.centers.unsqueeze(0), dim=2)
            d_sim = torch.mm(pnts, self.centers.T)
        """
        assert self.centers is not None, "Missing centers!"
        sim = torch.cdist(pnts, self.centers, p=2)
        # return sim

        # Compute silhouette-like similarity
        sim_sorted = torch.sort(sim, dim=-1, descending=False)[0]
        a = sim_sorted[:, 0]     # Cohesion
        b = sim_sorted[:, 1]     # Separation

        silhouette = (b - a) / torch.max(a, b)
        silhouette[torch.isnan(silhouette)] = 0.0  # Set NaN values to 0
        return silhouette

    def silhouette_scores(self, pnts):
        """
        Compute the silhouette score for points using the centers and saved
        samples.
        """
        # Compute similarities with centers
        sim = torch.cdist(pnts, self.centers, p=2)
        args = sim.argsort(-1)
        labels = args[:, 0]
        nn_center = args[:, 1]

        # Calculate the average distance of each sample to all saved samples
        # for its clusters
        a_sim = torch.cdist(pnts, self.samples[labels], p=2)
        a = a_sim.mean(-1)

        # Calculate the minimum average distance of the i-th sample to all
        # samples in a different cluster
        a_sim = torch.cdist(pnts, self.samples[nn_center], p=2)
        b = a_sim.mean(-1)

        # Calculate the silhouette score for each sample
        silhouette = (b - a) / torch.max(a, b)
        silhouette[torch.isnan(silhouette)] = 0.0  # Set NaN values to 0
        return silhouette

    def tune(self, pnts: torch.Tensor, preds: torch.Tensor, y: torch.Tensor):
        """
        Compute the confidence intervals for in-distribution data.
        Pnts should be the embeddings of train or validation data.

        Args:
            pnts: in-distribution embeddings. Shape (*, embed size).
            preds: logits or probability for each class (*, num classes)
            y: true label of embeddings. Shape (*,).
        Return:
            None
        """
        num_classes = preds.size(-1)
        y_preds = preds.argmax(-1)

        if self.centers is None:
            # Average of points as class center
            self.centers = torch.stack([
                pnts[torch.logical_and(y_preds == _c, y_preds == y)].mean(0)
                for _c in range(num_classes)
            ]).to(self.dev)

        # Compute similarity
        sim = self.__similarity__(pnts)

        # Find interval for in-distribution data
        for _c in range(num_classes):
            _mask = torch.logical_and(y_preds == _c, y_preds == y)
            # Silhouette
            self.ood_th.append(sim[_mask].quantile(self._q))

    def from_predictions(self, pnts, y_pred):
        """
        Compute OOD-prediction from model embeddings.

        Args:
            pnts:
            y_pred:
        Return:
        """
        # Compute similarity
        sim = self.__similarity__(pnts)
        size = pnts.size(0)

        #
        ood_pred = torch.zeros(size, dtype=torch.bool)
        for _c in torch.unique(y_pred, sorted=True):
            #
            _mask = y_pred == _c
            ood_pred[_mask] = sim[_mask] <= self.ood_th[_c.item()]

        return ood_pred.int()

    def predict(self, loader):
        """
        Generate model and OOD predictions.

        Args:
            loader: data loader.
        Return:
             - The model predictions.
             - The OOD predictions, true if OOD.
        """
        #
        preds, y_truth, z = predictions(self.model, loader, device=self.device)
        y_preds = preds.argmax(-1)
        #
        return y_preds, self.from_predictions(z, y_preds)
