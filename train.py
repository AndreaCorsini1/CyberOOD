import argparse
import torch
import FNN
from datasets.utils import dataset_info, make_dataset, make_loader
from sklearn.metrics import silhouette_samples as sil_scores
from sklearn.metrics import f1_score
from utils import performance, predictions, CenterLoss
from plot import plot_embeddings

#
PLOT_STEP = 300
SEED = 12345
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_model(model_name: str, in_size: int, num_classes: int):
    """
    Make the model and its training function.

    Args
        model_name: name of the model.
        in_size: number of features in a NetFlow
        num_classes: number of classes in the dataset.
    Return:
        Model and its training function.
    """
    model = FNN.FNN(num_classes=num_classes, in_size=in_size,
                    p1=0.3, p2=0.3, p3=0.3)
    if model_name.lower() == 'fnn':
        train_fn = train_fnn
    elif model_name.lower() == 'centerfnn':
        train_fn = train_center
        model.name = 'CenterFNN'
    else:
        raise RuntimeError(f"Unknown model {model_name}")
    return model.to(DEV), train_fn


def train_fnn(m, dl_train, dl_val,
              epochs: int = 20,
              lr_model: float = 0.001,
              weight: torch.Tensor = None,
              classes=('Benign', 'Malicious'),
              **kwargs):
    """
    Standard training function for a model.

    Args:
        m: model to train.
        dl_train: training loader.
        dl_val: validation loader.
        epochs: number of training epochs.
        lr_model: learning rate of the optimizer.
        weight: weight for each class, used inside the loss function.
        classes: name of each class for printing the output.
    Return:
        Trained model and None
    """
    # Optimizer and Loss function
    ce_loss = torch.nn.CrossEntropyLoss(weight=weight)
    opti_model = torch.optim.Adam(m.parameters(), lr=lr_model)

    #
    b_score = 0
    for e in range(epochs):
        m.train()
        _losses = []
        _silhouette, _cnt = 0, 0

        for x, y in dl_train:
            x = x.to(DEV)
            y = y.to(DEV)

            # PREDICT
            y_pred, z = m(x.to(DEV))

            # LOSS
            loss = ce_loss(y_pred, y)
            _losses.append(loss.item())
            if y.unique().size(0) > 1:
                _silhouette += sum(sil_scores(z.cpu().detach().numpy(),
                                              y.cpu().detach().numpy()))
                _cnt += z.size(0)

            # OPTIMIZE
            opti_model.zero_grad()
            loss.backward()
            opti_model.step()

        # Print stats
        sum_loss = sum(_losses)
        print(f'#############################################\n\tEPOCH {e:02}: '
              f'avg loss={sum_loss/len(_losses):.5f}, cum loss={sum_loss:.3f}')
        print(f"\t\tavg sil={_silhouette / _cnt:.5f}")

        # Validation
        val_pred, val_y, val_z = predictions(m, dl_val, device=DEV)
        if (e + 1) % PLOT_STEP == 0:
            plot_embeddings(val_z.numpy(), val_y, clf=m.clf.cpu(),
                            title=f"Epoch {e}", **kwargs)
            m = m.to(DEV)
        val_y = val_y.cpu()
        y_pred = val_pred.cpu().argmax(-1)
        performance(val_y, y_pred, classes=classes)
        f1_s = f1_score(val_y.numpy(), y_pred.numpy(), average='macro')
        if b_score < f1_s:
            print(f"\tNew Best F1: {f1_s:.5f}")
            b_score = f1_s
            torch.save(m, f'{base_path}/{m.name}_{args.scenario}_'
                          f'{"bin" if args.binary else "mc"}.pt')
        # Early stopping
        if sum_loss / len(_losses) < 0.0001:
            print(f"Early stopping at EPOCH {e}")
            break
    #
    return m, None


def train_center(m, dl_train, dl_val,
                 epochs: int = 20,
                 lr_model: float = 0.001,
                 lr_loss: float = 0.0001,
                 alpha: float = 1,
                 weight: torch.Tensor = None,
                 classes=('Benign', 'Malicious'),
                 num_classes: int = 2,
                 **kwargs):
    """
    Training function using Center Loss as contrastive signal.

    Args:
        m: model to train.
        dl_train: training loader.
        dl_val: validation loader.
        epochs: number of training epochs.
        lr_loss: learning loss for updating centers.
        lr_model: learning rate of the optimizer.
        alpha: factor for balancing the influence of center loss.
        weight: weight for each class, used inside the loss function.
        classes: name of each class for printing the output.
    Return:
        Trained model and generated center at the end of training.
    """
    # Loss function
    ce_loss = torch.nn.CrossEntropyLoss(weight=weight)
    center_loss = CenterLoss(num_classes=num_classes, device=DEV)
    # Optimizer
    opti_model = torch.optim.Adam(m.parameters(), lr=lr_model)
    opti_loss = torch.optim.Adam(center_loss.parameters(), lr=lr_loss)

    #
    b_score = 0
    for e in range(epochs):
        m.train()
        _losses = []
        _silhouette, _cnt = 0, 0

        for x, y in dl_train:
            x = x.to(DEV)
            y = y.to(DEV)

            # PREDICT
            y_pred, z = m(x)

            # LOSS
            loss = ce_loss(y_pred, y)
            # Apply center loss only to correct predictions
            _mask = y == y_pred.argmax(-1)
            if any(_mask):
                loss += center_loss(z[_mask], y[_mask]) * alpha
            _losses.append(loss.item())
            if y.unique().size(0) > 1:
                _silhouette += sum(sil_scores(z.cpu().detach().numpy(),
                                              y.cpu().detach().numpy()))
                _cnt += z.size(0)

            # OPTIMIZE
            opti_loss.zero_grad()
            opti_model.zero_grad()
            loss.backward()
            opti_model.step()
            # multiple (1. / alpha) to remove the effect of alpha
            for param in center_loss.parameters():
                param.grad.data *= (1. / alpha)
            opti_loss.step()

        # Print stats
        sum_loss = sum(_losses)
        print(f'#############################################\n\tEPOCH {e:02}: '
              f'avg loss={sum_loss/len(_losses):.5f}, cum loss={sum_loss:.3f}')
        print(f"\t\tavg sil={_silhouette / _cnt:.5f}")

        # Validation
        val_pred, val_y, val_z = predictions(m, dl_val, device=DEV)
        if (e + 1) % PLOT_STEP == 0:
            plot_embeddings(val_z.numpy(), val_y, clf=m.clf,
                            title=f"Epoch {e}", **kwargs)
        y_pred = val_pred.argmax(-1)
        performance(val_y, y_pred, classes=classes)
        #
        f1_s = f1_score(val_y.numpy(), y_pred.numpy(), average='macro')
        if b_score < f1_s:
            print(f"\tNew Best F1: {f1_s:.5f}")
            b_score = f1_s
            torch.save(m, f'{base_path}/{m.name}_{args.scenario}_'
                          f'{"bin" if args.binary else "mc"}.pt')
        # Early stopping
        if sum_loss / len(_losses) < 0.001:
            print(f"Early stopping at EPOCH {e}")
            break
    #
    return m, center_loss.centers.detach()


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument("-dataset", type=str, default='ids2021',
                    required=False, help="Name of the dataset to load.")
parser.add_argument("-model", type=str, default='centerfnn',
                    required=False, help="Name of the model.")
parser.add_argument("-scenario", type=str, default='Q1',
                    required=False, help="Training scenario (which attacks).")
parser.add_argument("-epochs", type=int, default=25,
                    required=False, help="Number of epochs.")
parser.add_argument("-batch_size", type=int, default=512,
                    required=False, help="Batch size.")
parser.add_argument("-lr", type=float, default=0.0005,
                    required=False, help="Learning rate.")
parser.add_argument("-binary", type=bool, default=False,
                    required=False, help="Binary training.")
args = parser.parse_args()


if __name__ == '__main__':
    print(f"Running on {DEV}", '\n', args)
    # fix random seeds
    torch.manual_seed(SEED)

    # Make dataset
    print(f'Load and make the datasets for {args.scenario}:')
    d_path, classes, scenarios, mapping, _ = dataset_info(args.dataset)
    train_s, val_s, _, _ = make_dataset(d_path, scenarios[args.scenario],
                                        verbose=True)
    if args.binary:
        train_s = (train_s[0], (train_s[1] > 0).long())
        val_s = (val_s[0], (val_s[1] > 0).long())

    # Make loaders
    print('Make loaders:')
    dl_train, _, enc_train = make_loader(*train_s, args.batch_size,
                                         balance_sampling=True)
    dl_val, _, _ = make_loader(*val_s, args.batch_size, shuffle=False,
                               encoder=enc_train)

    # Make model
    n_classes = enc_train.classes_.shape[0]
    model_, train_fn_ = make_model(args.model,
                                   in_size=train_s[0].size(-1),
                                   num_classes=n_classes)
    print(model_)

    # Train
    base_path = f"./checkpoints/{args.dataset}"
    print(f"Training {args.model} ...")
    model, centers = train_fn_(model_, dl_train, dl_val,
                               epochs=args.epochs,
                               num_classes=n_classes,
                               classes=tuple(mapping[c] for c in enc_train.classes_),
                               alpha=2,
                               mapping=mapping,
                               label_encoder=enc_train)
