import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from torch.utils.data import WeightedRandomSampler


def dataset_info(data_name: str = 'ids2018'):
    """
    Load basic information about dataset.

    Args:
        data_name: name of the dataset to load info for.
    Return:
        ...
    """
    if data_name.lower() == 'ids2017':
        d_path = 'datasets/IDS2017/'
        classes = [
            "Benign", "FTP-Patator", "SSH-Patator", "DoS GoldenEye",
            "DoS Hulk", "DoS Slowhttp", "DoS slowloris",
            "Heartbleed", "Web-BForce", "Web Attack - XSS",
            "Web-Sql Injection", "Infiltration",
            "Bot", "PortScan", "DDoS"
        ]
        # Training scenarios
        scenarios = {
            'B': ['benign'],
            'OOD': ['heartbleed', 'web attack - xss', 'web-sql injection', 'infiltration'],
            'C2': ['benign', 'dos hulk', 'portscan', 'ssh-patator'],
            'S2': ['benign', 'ddos', 'ssh-patator', 'dos slowloris'],
        }
        # Drop: Heartbleed, Sql Injection, XSS and Infiltration
        drop_attacks = [7, 9, 10, 11]
    elif data_name.lower() == 'ids2018':
        d_path = 'datasets/IDS2018/'
        classes = [
            'Benign', 'FTP-BForce', 'SSH-BForce',
            'DoS-GoldenEye', 'DoS-Hulk', 'DoS-SlowHTTP', 'DoS-Slowloris',
            'Web-BForce', 'XSS-BForce', 'SQL Injection',
            'Infiltration', 'Bot',
            'DDOS-HTTP', 'DDOS-UDP', 'DDOS-HOIC'
        ]
        scenarios = {
            'B': ['benign'],
            'OOD': ['sql injection', 'infiltration'],
            'C2': ['benign', 'dos-hulk', 'ftp-bforce', 'ddos-hoic'],
            'M2': ['benign', 'dos-goldeneye', 'bot', 'ddos-udp'],
            'R2': ['benign', 'dos-hulk', 'ssh-bforce', 'ddos-http'],
        }
        # Drop attacks: Sql Injection
        drop_attacks = [9, 10]
    else:
        raise RuntimeError(f"Unknown dataset {data_name}")

    # Mapping to names
    mapping = dict(zip(range(len(classes)), classes))
    return d_path, classes, scenarios, mapping, drop_attacks


def make_dataset(dataset_path, train_names, val_split=0.15, seed=12345,
                 scaler=None, verbose=0):
    """

    Args:
        dataset_path:
        train_names:
        val_split:
        seed:
        verbose:
    :return:
    """
    # plot_distribution(dataset_path)
    # Load the chunks
    train_x, train_y, test_x, test_y = [], [], [], []
    b_path = os.path.join(dataset_path, 'chunks')
    for f in os.listdir(b_path):
        file_p = os.path.join(b_path, f)
        if os.path.isfile(file_p):
            name = f.rsplit('-', 1)[0]
            x, y = torch.load(file_p)
            if any([name in tn for tn in train_names]):
                train_x.append(x)
                train_y.append(y)
            else:
                test_x.append(x)
                test_y.append(y)

    # Split train/val/test
    train_x = torch.cat(train_x, dim=0).numpy()
    train_y = torch.cat(train_y, dim=0).numpy()
    test_x, test_y = torch.cat(test_x, dim=0), torch.cat(test_y, dim=0)
    # Add constant and log scale non-boolean features
    train_x[:, :-2] = np.log(train_x[:, :-2] + 1.)
    test_x[:, :-2] = np.log(test_x[:, :-2] + 1.)
    #
    x_t, x_v, y_t, y_v = train_test_split(train_x, train_y,
        test_size=val_split, stratify=train_y, shuffle=True, random_state=seed)
    train_x, train_y = torch.from_numpy(x_t), torch.from_numpy(y_t)
    val_x, val_y = torch.from_numpy(x_v), torch.from_numpy(y_v)

    #
    if verbose:
        print(f'  Train shape={train_x.shape}:\n\t', np.unique(train_y, return_counts=True))
        print(f'  Val shape={val_x.shape}:\n\t', np.unique(val_y, return_counts=True))
        print(f'  Test shape={test_x.shape}:\n\t', np.unique(test_y, return_counts=True))

    # Scale
    if scaler is None:
        # scaler = StandardScaler()
        scaler = RobustScaler()
        train_x = torch.from_numpy(scaler.fit_transform(train_x.numpy()))
    else:
        train_x = torch.from_numpy(scaler.transform(train_x.numpy()))
    val_x = torch.from_numpy(scaler.transform(val_x.numpy()))
    test_x = torch.from_numpy(scaler.transform(test_x.numpy()))

    return (train_x, train_y), (val_x, val_y), (test_x, test_y), scaler


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int,
                shuffle: bool = True,
                encoder: LabelEncoder = None,
                balance_sampling: bool = False,
                verbose: bool = False):
    """
    Helper for making data loader.
    """
    if encoder is None:
        encoder = LabelEncoder()
        _y = encoder.fit_transform(y.numpy())
    else:
        _y = encoder.transform(y.numpy())
    unique_y, counts = np.unique(_y, return_counts=True)

    #
    tensor_x = x.type(torch.float32)
    tensor_y = torch.from_numpy(_y).type(torch.long)
    weights = torch.from_numpy(counts).type(torch.float32)
    if verbose:
        print("\tClasses:", encoder.classes_)
        print("\tNum:", counts)

    # Make dataloader
    ds = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    if balance_sampling:
        #
        _weights = torch.max(weights) / weights
        sampler = WeightedRandomSampler(weights=_weights[tensor_y],
                                        num_samples=len(ds), replacement=True)
        return torch.utils.data.DataLoader(
            ds, sampler=sampler, batch_size=batch_size), weights, encoder
    else:
        return torch.utils.data.DataLoader(
            ds, shuffle=shuffle, batch_size=batch_size), weights, encoder


def make_chunks(df: pd.DataFrame, classes: list, start_from: int = 0,
                drop_attr: tuple = ('label', 'Timestamp', 'date')):
    """
    Make chunks of a dataset by splitting on each attack type per date.

    Args:
        df: (DataFrame)
            The dataset
        classes: (list of str)
            Name of the classes. It is used to transform an index into a
            class name.
        start_from: (int)
        drop_attr: (tuple)
            Attributes to remove.
    """
    # Remove the benign traffic: we suppose that each attack has its own label
    mal_seq = df.label[df.label != 0]
    change_pnts = mal_seq[mal_seq.shift(-1) != mal_seq].index.tolist()
    if 0 not in change_pnts:
        change_pnts = [start_from - 1] + change_pnts
    _intervals = [(s+1, e) for s, e in zip(change_pnts[:-1], change_pnts[1:])]

    #
    intervals = []
    for s, e in _intervals:
        c = df.iloc[s:e]
        attacks = c.label.unique()
        if len(attacks) > 2:
            raise RuntimeError(f"More than two labels in one chunk {attacks}!")
        l = attacks[1] if attacks[0] == 0 else attacks[0]
        l_min, l_max = c[c.label == l].index.min(), c[c.label == l].index.max()
        assert len(df.iloc[s:l_min].label.unique()) == 1, "Err1"
        intervals.append((s, l_min))
        intervals.append((l_min, l_max))

    # Split the dataset into chunks
    chunks = {}
    repeated = {}
    for c_idx, (s, e) in enumerate(intervals):
        attacks = df.label.iloc[s:e].unique()
        # Make a consistent name
        name = f"{classes[attacks[0]].lower()}-001" if len(attacks) == 1 else \
            f"{classes[attacks.max()].lower()}-001"
        if name in chunks:
            i = repeated.get(name, 2)
            repeated[name] = i + 1
            name = f"{name.rsplit('-', 1)[0]}-{i:03}"
        #
        chunks[name] = (
            df[s:e].drop([*drop_attr], axis=1).values,   # X
            df[s:e].label.values,                        # Y
        )
    return chunks


#
FEATURES = [
    'flow_duration', 'flow_pkts/s',
    'flow_iat_mean', 'flow_iat_std',
    'idle_mean', 'active_mean',
    'subflow_fwd_byts', 'subflow_bwd_byts',
    'fwd_pkts/s', 'bwd_pkts/s',
    'fwd_seg_size_min',
    'tot_fwd_pkts', 'tot_bwd_pkts',
    'fwd_pkt_len_max', 'bwd_pkt_len_max',
    'ack_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt',
    'dst_regport', 'dst_wkport',
    'label'
]


if __name__ == '__main__':
    # This is for preparing the datasets for training and testing
    # name = "CICIDS2017"
    name = "CICIDS2018"
    out_path = f'datasets/{name}/chunks/'

    #
    d_path, classes, scenarios, mapping, drop_attacks = dataset_info(name)

    #
    print(f"Loading data {d_path} ...")
    data = pd.read_csv(d_path + 'data.csv.zip')[FEATURES + ['Timestamp']]
    data = data[~data.isna().any(axis=1)]  # Drop Na
    # print(data.shape)
    # print(data.label.value_counts())

    #
    if drop_attacks is not None:
        data = data.drop(data.index[data.label.isin(drop_attacks)])\
            .reset_index(drop=True)

    # Add the date to split the attack by day
    data['date'] = pd.to_datetime(data.Timestamp).dt.date
    chunks = make_chunks(data, classes)
    for k, v in chunks.items():
        print(f"  Attack {k}:")
        print(f"  \tProp mal: {sum(v[1] > 0) / len(v[1]):.5f}")
        print(f"  \tNum benign: {sum(v[1] == 0)}")
        #
        torch.save((torch.from_numpy(np.float32(v[0])),
                    torch.from_numpy(np.int32(v[1]))), f"{out_path}{k}.pt")
