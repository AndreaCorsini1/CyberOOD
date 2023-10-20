import argparse
import torch
import pandas as pd
from datasets.utils import dataset_info, make_loader, make_dataset
from utils import predictions
from OOD import confidence, distance, gradient
from sklearn import metrics as skl_metrics


#
avoid = ['accuracy', 'macro avg', 'weighted avg']


def label_remapping(preds, y_truth):
    # Find OOD predictions and mask the correct ones
    _pred = preds > 0
    _mask = _pred & (y_truth > 0)
    # Transform correct predictions into the original labels
    y_pred = _pred.int()
    y_pred[_mask] = y_truth[_mask].int()

    return y_pred


def save_results(ood, comb, out_path):
    data = {
        k: [v['recall'], v['precision'], v['f1-score'], v['support']]
        for k, v in ood.items() if k not in avoid
    }
    # Combined predictions
    for k, v in comb.items():
        if k not in avoid:
            data[f'c_{k}'] = [v['recall'], v['precision'],
                              v['f1-score'], v['support']]
    pd.DataFrame.from_dict(data, orient='index',
                           columns=['R', 'P', 'F1', 'Sup']).to_csv(out_path)


def compress(scenario, bin: bool = True):
    data = {}
    for t in ['conf', 'mcd', 'odin', 'mal', 'sim', 'ens']:
        df = pd.read_csv(f"{scenario}_{t}.csv", index_col=0)
        data[(t, 'R')] = df['R']
        data[(t, 'P')] = df['P']
        data[(t, 'F1')] = df['F1']
    pd.DataFrame(data).to_csv(f"{scenario}_tot.csv")
    #
    if bin:
        data = {}
        for t in ['conf', 'mcd', 'odin', 'mal', 'sim', 'ens']:
            df = pd.read_csv(f"{scenario}_{t}_bin.csv", index_col=0)
            data[(t, 'R')] = df['R']
            data[(t, 'P')] = df['P']
            data[(t, 'F1')] = df['F1']
        pd.DataFrame(data).to_csv(f"{scenario}_tot_bin.csv")


def _test(y_pred, ood_pred, y_truth, out_path, classes):
    # MULTI-CLASS how much traffic is rejected
    _pred = label_remapping(ood_pred, y_truth)
    ood = skl_metrics.classification_report(y_truth.numpy(), _pred.numpy(),
            output_dict=True, zero_division=0)  # , target_names=classes)
    # print(ood)
    # COMBINED how much malicious traffic is labeled as malicious
    com_pred = (y_pred > 0).int()
    com_pred[ood_pred > 0] = 1
    com_pred = label_remapping(com_pred, y_truth)
    com = skl_metrics.classification_report(y_truth.numpy(), com_pred.numpy(),
            output_dict=True, zero_division=0)  # , target_names=classes)
    # print(com)
    #
    if out_path is not None:
        save_results(ood, com, out_path + '.csv')

    # BINARY
    _bin = (y_truth > 0).int().numpy()
    ood_bin = skl_metrics.classification_report(_bin, (_pred > 0).int().numpy(),
            output_dict=True, zero_division=0, target_names=['Ben', 'Mal'])
    # print(ood_bin)
    org_bin = skl_metrics.classification_report(_bin, (y_pred > 0).int().numpy(),
            output_dict=True, zero_division=0, target_names=['Ben', 'Mal'])
    # print(org_bin)
    # Save in file
    if out_path is not None:
        save_results(ood_bin, org_bin, out_path + '_bin.csv')

    return com_pred, _pred


def mcd_test(m, dl_val, dl_test, classes: list,
             out_path: str = None, temp: float = 2):
    print("\nMONTE CARLO DROPOUT")
    y_truth = dl_test.dataset.tensors[1]

    #
    mcd = confidence.MCDropout(m, temp=temp)
    mcd.tune(dl_val)
    y_pred, ood_preds = mcd.predict(dl_test)
    ood_pred = (ood_preds > 0).any(-1).int()

    #
    return _test(y_pred, ood_pred, y_truth, out_path, classes)


def knn_test(m, dl_train, dl_val, dl_test, classes: list, out_path: str = None):
    print("\nKNN ...")
    y_truth = dl_test.dataset.tensors[1]

    #
    knn = distance.KNNDetector(m, k=25, leaf_size=50)
    knn.tune(dl_train, dl_val)
    y_pred, ood_preds = knn.predict(dl_test)
    ood_pred = (ood_preds > 0).any(-1).int()

    #
    return _test(y_pred, ood_pred, y_truth, out_path, classes)


def conf_test(m, dl_val, dl_test, classes: list,
              out_path: str = None, temp: float = 2):
    print("\nCONFIDENCE")
    y_truth = dl_test.dataset.tensors[1]

    #
    conf = confidence.Confidence(m, temp=temp)
    conf.tune(dl_val)
    y_pred, ood_pred = conf.predict(dl_test)

    #
    return _test(y_pred, ood_pred, y_truth, out_path, classes)


def odin_test(m: torch.nn.Module,
              val: torch.utils.data.DataLoader,
              test: torch.utils.data.DataLoader,
              ood: torch.utils.data.DataLoader,
              classes: list,
              out_path: str = None,
              temp: float = 2):
    print("\nODIN")
    # Tune ODIN
    odin = gradient.ODIN(m, temp=temp)
    odin.tune(val, ood)

    #
    y_true = test.dataset.tensors[1]
    y_pred, ood_pred = odin.predict(test)
    return _test(y_pred, ood_pred, y_true, out_path, classes)


def mal_test(m: torch.nn.Module,
             train: torch.utils.data.DataLoader,
             val: torch.utils.data.DataLoader,
             test: torch.utils.data.DataLoader,
             ood: torch.utils.data.DataLoader,
             classes: list,
             out_path: str = None):
    print("\nMahalanobis")

    # Tune
    mahal = gradient.Mahalanobis(m)
    _, y_train, z = predictions(m, train)
    mahal.tune(val, ood, z, y_train)

    #
    y_true = test.dataset.tensors[1]
    y_pred, ood_pred = mahal.predict(test)
    return _test(y_pred, ood_pred, y_true, out_path, classes)


def sim_test(m, dl_val, dl_test, classes: list, out_path: str = None):
    print("\nSIMILARITY")
    y_truth = dl_test.dataset.tensors[1]

    #
    sim = distance.SilhoDetector(m)
    preds, y_val, z = predictions(m, dl_val)
    sim.tune(z, preds, y_val)
    y_pred, ood_pred = sim.predict(dl_test)

    #
    return _test(y_pred, ood_pred, y_truth, out_path, classes)


#
parser = argparse.ArgumentParser(description='Test OOD')
parser.add_argument("-dataset", type=str, default='ids2018',
                    required=False, help="Name of the dataset to load.")
parser.add_argument("-ood_dataset", type=str, default='ids2017',
                    required=False, help="Name of the OOD dataset.")
parser.add_argument("-model", type=str, default='centerfnn',
                    required=False, help="Path to the model.")
parser.add_argument("-scenario", type=str, default='C2',
                    required=False, help="Testing scenario (which attacks).")
parser.add_argument("-batch", type=int, default=1024,
                    required=False, help="Bach size.")
parser.add_argument("-binary", type=bool, default=False,
                    required=False, help="Binary training.")
args = parser.parse_args()


if __name__ == '__main__':
    ######
    print(f'Load and make the datasets for {args.scenario}:')
    d_path, classes, scenarios, mapping, drop_attacks = dataset_info(args.dataset)
    train_s, val_s, test_s, scaler = make_dataset(d_path,
                                                  scenarios[args.scenario],
                                                  verbose=True)

    # Make the data loader for the testing scenario
    if args.binary:
        train_s = (train_s[0], (train_s[1] > 0).long())
        val_s = (val_s[0], (val_s[1] > 0).long())
    dl_train, _, enc_train = make_loader(*train_s, batch_size=args.batch,
                                         shuffle=False)
    dl_val, _, _ = make_loader(*val_s, batch_size=args.batch, shuffle=False)

    # OOD attacks
    _mask = test_s[1] > 0
    dl_test, _, enc_test = make_loader(test_s[0][_mask], test_s[1][_mask],
                                       batch_size=args.batch, shuffle=False)
    classes_ = [tuple(mapping[c] for c in enc_test.classes_)]

    ###### Data for tuning
    _mask = torch.zeros(test_s[1].size(), dtype=torch.bool)
    for ood_y in drop_attacks:
        _mask[test_s[1] == ood_y] = True
    dl_ood, _, enc_ood = make_loader(test_s[0][_mask], test_s[1][_mask],
                                     batch_size=args.batch, shuffle=False)

    # Load the model
    base_path = f"./checkpoints/{args.dataset}"
    model_ = torch.load(f'{base_path}/{args.model}_{args.scenario}_'
                        f'{"bin" if args.binary else "mc"}.pt')
    print(model_)

    ###### TEST DETECTORS
    out_path = f"output/{args.dataset.lower()}/{args.model.lower()}" \
               f"/{args.scenario.lower()}_{'bin' if args.binary else 'mc'}"

    conf_com, conf_ood = conf_test(model_, dl_val, dl_test, classes_,
                                   out_path + "_conf")
    mcd_com, mcd_ood = mcd_test(model_, dl_val, dl_test, classes_,
                                out_path + "_mcd")
    knn_com, knn_ood = knn_test(model_, dl_train, dl_val, dl_test, classes_,
                                out_path + "_knn")
    sim_com, sim_ood = sim_test(model_, dl_train, dl_test, classes_,
                                out_path + "_sim")
    odin_com, odin_ood = odin_test(model_, dl_val, dl_test, dl_ood,
                                   classes_, out_path + "_odin")
    mal_com, mal_ood = mal_test(model_, dl_train, dl_val, dl_test, dl_ood,
                                classes_, out_path + "_mal")

    y_t = dl_test.dataset.tensors[1]
    _tmp = torch.stack([conf_ood, mcd_ood, odin_ood, mal_ood,
                        knn_ood, sim_ood, y_t], dim=1)
    pd.DataFrame(
        _tmp.numpy(),
        columns=["CONF", "MCD", "ODIN", "MD", "KNN", "SIM", "Y"]
    ).to_csv(out_path + "_tot.csv", index=False)
