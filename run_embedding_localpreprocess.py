"""
- Run a simple clustering on input features and calculate score of the features
"""

import numpy as np
import argparse
import os
import pandas as pd

from sklearn.cluster import AffinityPropagation
from sklearn import metrics

from sklearn.cluster import MiniBatchKMeans, KMeans
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import torch
import sys
from models import Autoencoder

from local_utils import get_facies_mapping

from run_embedding_geotut import train_model_autoencoder

import utils

TARGET_COLUMN = 'Facies_Full PETREL'
DELTA_DEPTH = 0.5

# Should get len(TRAIN_COLUMNS) + 3 columns: WELL, DEPTH, TARGET_COLUMN
TRAIN_COLUMNS = ['CAL', 'GR', 'NPHI-3 TNPH', 'LLD', 'RHOB',
                 'LLS', 'PEF-4PE', 'Perm', 'PHIE', 'PHIT',
                 'SW', 'SWT']


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
    filename = os.path.join(base_folder, 'preprocess_full_5wells.csv')
    df_train = pd.read_csv(filename)
    df_train['WELL'] = df_train['WELL'].astype('category')
    print(df_train.columns)
    print(df_train['WELL'].unique())

    return df_train


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


def train_model(args):
    model = args.model
    base_folder = args.base_folder
    n_d = 32
    n_a = 32
    n_steps = 5
    # gamma = args.gamma

    utils.seed_everything(args.seed)
    log_folder = utils.create_log_folder("./output", "clustering_local_%s" % model, sys.argv)

    output_file = open('%s/log.txt' % log_folder, 'w')

    # Read data
    data = read_data(base_folder, add_stats_fts=args.use_seq_fts)
    print(data.shape)

    df, facies_labels = get_facies_mapping(data, TARGET_COLUMN)
    print(df[TARGET_COLUMN].unique())
    print(facies_labels)
    n_classes = 9

    X_train = df.drop(['DEPTH', TARGET_COLUMN, 'X_Axis', 'Y_Axis', 'WELL'], axis=1)
    y_train = df[TARGET_COLUMN].values
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
        model = Autoencoder(in_shape=X_train.shape[1], enc_shape=16)

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
            pretraining_ratio=0.75,
            max_epochs=200,
        )

        _, embedded_X = unsupervised_model.predict(X_train)
        np.savez('%s/embedded_output.npz' % log_folder, embedded_X=embedded_X, labels=labels_true)
        run_evaluate_embedding(embeded_X=embedded_X, labels_true=labels_true, n_clusters=n_classes)


if __name__ == "__main__":
    args = get_parser().parse_args()
    train_model(args)
