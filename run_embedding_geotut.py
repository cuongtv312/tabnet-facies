"""
- Run a simple clustering on input features and calculate score of the features
"""

import numpy as np
import argparse
import os
import pandas as pd

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader, TensorDataset


from sklearn.cluster import MiniBatchKMeans, KMeans
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import torch
import sys
from models import Autoencoder


from local_utils import get_facies_mapping

import utils

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D', 'PS', 'BS']
feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']

class WelllogDataset(Dataset):
    def __init__(self, X, y, seq_length):
        """
        Args:
        """
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.data_length = len(X)

        self.metrics = self.create_xy_pairs()

    def create_xy_pairs(self):

        pairs = []
        for idx in range(self.data_length - self.seq_length):
            x = self.X[idx:idx + self.seq_length]
            y = self.y[idx:idx + self.seq_length]
            # y = y - 1
            pairs.append((x, y))
        return pairs

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.metrics[idx]


def get_parser():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--base_folder', type=str, default='./input',
                        help='Base input folder')
    parser.add_argument('--seed', type=int, default=2022,
                        help='Running seed')
    parser.add_argument('--use_seq_fts', default=False, action='store_true')

    parser.add_argument('--model', type=str, default='default', choices=['default', 'mlp', 'tabnet'])
    parser.add_argument('--cluster', type=str, default='knn', choices=['knn', 'dbscan'])

    return parser


# Read and preprocessing data
def read_data(base_folder, add_stats_fts=False):
    print('Read input data')
    filename = os.path.join(base_folder, 'facies_vectors.csv')
    df_train = pd.read_csv(filename)
    df_train['Well Name'] = df_train['Well Name'].astype('category')
    df_train['Formation'] = df_train['Formation'].astype('category')
    print(df_train.head())
    print(df_train.shape)

    print('Read depth data')
    df_depth = pd.read_csv(os.path.join(base_folder, 'prediction_depths.csv'))
    df_depth.set_index(["Well Name", "Depth"], inplace=True)
    print(df_depth.head())
    print(df_depth.shape)

    # Print read predict data
    df_test = pd.read_csv(os.path.join(base_folder, 'blind_stuart_crawford_core_facies.csv'))
    df_test.rename(columns={'Depth.ft': 'Depth'}, inplace=True)
    df_test.rename(columns={'WellName': 'Well Name'}, inplace=True)

    print(df_test.head())

    test_data = pd.read_csv(os.path.join(base_folder,'validation_data_nofacies.csv'))
    well_ts = test_data['Well Name'].values
    depth_ts = test_data['Depth'].values
    y_ts = []
    prev = 0
    for i in range(len(well_ts)):
        tmp = df_test[(df_test['Well Name'] == well_ts[i]) & (df_test['Depth'] == depth_ts[i])]['LithCode']

        if len(tmp) > 0:
            y_ts += [tmp.values[0]]
            prev = tmp.values[0]
        else:
            y_ts += [prev]

    test_data['Facies'] = y_ts
    return df_train, test_data

def run_evaluate_embedding(embeded_X, labels_true, n_clusters):
    mbk = MiniBatchKMeans(
        init="k-means++",
        n_clusters= n_clusters,
        batch_size=128,
        n_init=10,
        max_no_improvement=10,
        verbose=0,
    )

    af = mbk.fit(embeded_X)
    cluster_centers_ = mbk.cluster_centers_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Homogeneity: %0.4f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.4f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.4f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.4f" % metrics.adjusted_rand_score(labels_true, labels))
    print(
        "Adjusted Mutual Information: %0.4f"
        % metrics.adjusted_mutual_info_score(labels_true, labels)
    )
    print(
        "Silhouette Coefficient: %0.4f"
        % metrics.silhouette_score(embeded_X, labels, metric="sqeuclidean")
    )


def train_model_autoencoder(model, X_train, device, seed=2020):
    p = np.random.permutation(seed)
    N = len(X_train)
    n_train = int(0.8 * N)
    X = X_train[p[:n_train]]
    eval_X = X_train[p[n_train:]]

    params = {'batch_size': 64,
              'shuffle': False,
              'drop_last': True,  # Disregard last incomplete batch
              'num_workers': 2}

    params_test = {'batch_size': 64,
                   'shuffle': False,
                   'drop_last': True,  # Disregard last incomplete batch
                   'num_workers': 2}

    training_ds = WelllogDataset(X, X, 1)
    training_dl = DataLoader(training_ds, **params)


    training_ds = WelllogDataset(X, X, 1)
    training_dl = DataLoader(training_ds, **params)

    test_ds = WelllogDataset(eval_X, eval_X, 1)
    test_dl = DataLoader(test_ds, **params_test)

    model = model.to(device)
    criterion = torch.nn.MSELoss()

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))

    # Train
    print("Training {} using {} started with total epoch of {}.".format(model.__class__.__name__, "Adam",
                                                                        50))

    for epoch in range(50):
        for _, (x, label) in enumerate(training_dl):

            x = x.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)

            pred, _ = model(x)
            train_loss = criterion(pred, label)

            # print(label, pred)
            optim.zero_grad()
            train_loss.backward()
            optim.step()


    valid_dataset = TensorDataset(torch.FloatTensor(X_train),
                                 torch.FloatTensor(X_train))
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=64,
                                  shuffle=False)

    output_nn = []
    model.eval()
    for i, (x, y) in enumerate(valid_dataloader):
        x = x.to(device)

        _ , output_y = model(x)
        output_nn += output_y.cpu().detach().numpy().tolist()

    return model, np.array(output_nn)


def train_model(args):
    model = args.model
    base_folder = args.base_folder
    n_d = 64
    n_a = 64
    n_steps = 5
    # gamma = args.gamma

    use_features = feature_names
    n_classes = len(facies_labels)

    utils.seed_everything(args.seed)
    log_folder = utils.create_log_folder("./output", "clustering_ks_%s" % model, sys.argv)

    output_file = open('%s/log.txt' % log_folder, 'w')

    # Read data
    data, test_data = read_data(base_folder)
    all_X = data.drop(['Facies', 'Formation', 'Depth'], axis=1)
    all_Y = data['Facies'] - 1
    test_Y = test_data['Facies'].values - 1

    train_X = all_X[use_features].values
    test_X = test_data[use_features].values

    X_train = np.vstack([train_X, test_X])
    y_train = np.hstack([all_Y, test_Y])

    print(data.shape)
    labels_true = y_train

    # Preprocess with SimpleImputer and MinMaxScaler
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = imp.fit_transform(X_train)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    if model == 'default':
        run_evaluate_embedding(embeded_X=X_train, labels_true=labels_true, n_clusters=n_classes)
    elif model == 'mlp':

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Autoencoder(in_shape=7, enc_shape=16)

        _, embedded_X = train_model_autoencoder(model, X_train, device, args.seed)
        run_evaluate_embedding(embeded_X=embedded_X, labels_true=labels_true, n_clusters=len(facies_labels))

    elif model == 'tabnet':
        p = np.random.permutation(args.seed + 1)
        N = len(X_train)
        n_train = int(0.8*N)

        unsupervised_model = TabNetPretrainer(
            n_d=n_d, n_a=n_a,
            n_steps=n_steps,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=1e-3),
            mask_type='entmax'  # 'entmax' or 'sparsemax'
        )

        unsupervised_model.fit(
            X_train=X_train[p[:n_train]],
            eval_set=X_train[p[n_train:]],
            pretraining_ratio=0.8,
            max_epochs=10,
        )

        _, embedded_X = unsupervised_model.predict(X_train)
        np.savez('%s/embedded_output.npz' % log_folder, embedded_X=embedded_X, labels=labels_true)
        run_evaluate_embedding(embeded_X=embedded_X, labels_true=labels_true, n_clusters=len(facies_labels))


if __name__ == "__main__":
    args = get_parser().parse_args()
    train_model(args)
