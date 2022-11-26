"""
Testing the tabnet with default features


"""

import numpy as np
import argparse

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import torch
import sys

from geotut_utils import accuracy, display_cm
from local_utils import get_facies_mapping
from run_xgb_localpreprocess import read_data
import utils

TARGET_COLUMN = 'Facies_Full PETREL'
DELTA_DEPTH = 0.5

# Should get len(TRAIN_COLUMNS) + 3 columns: WELL, DEPTH, TARGET_COLUMN
TRAIN_COLUMNS = ['CAL', 'GR', 'NPHI-3 TNPH', 'LLD', 'RHOB',
                 'LLS', 'PEF-4PE', 'Perm', 'PHIE', 'PHIT',
                 'SW', 'SWT']
#TRAIN_COLUMNS = ['CAL', 'GR-3,5 ECGR', 'NPHI-3 TNPH', 'RHOB',
#                 'Perm', 'PHIE', 'PHIT',
#                 'SW', 'SWT']

def get_parser():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--base_folder', type=str, default='./input',
                        help='Base input folder')
    parser.add_argument('--test_well_names', type=str, default='all',
                        help='all|102-HD-1X|103-DL-1X|103-DL-1X')
    parser.add_argument('--gamma', type=float, default=1.3,
                        help='Gamma value for tabnet')
    parser.add_argument('--seed', type=int, default=2022,
                        help='Running seed')
    parser.add_argument('--use_seq_fts', default=False, action='store_true')
    parser.add_argument('--use_pretrain', default=False, action='store_true')

    return parser


def train_model(args):
    utils.seed_everything(args.seed)
    log_folder = utils.create_log_folder("./output", 'tabnet_localdata', sys.argv)

    output_file = open('%s/log.txt' % log_folder, 'w')

    base_folder = args.base_folder
    test_well_names = args.test_well_names
    use_pretrain = args.use_pretrain
    n_d = 32
    n_a = 32
    n_steps = 5
    gamma = args.gamma

    data = read_data(base_folder, add_stats_fts=args.use_seq_fts)
    print(data.shape)

    df, facies_labels = get_facies_mapping(data, TARGET_COLUMN)
    print(df[TARGET_COLUMN].unique())
    print(facies_labels)
    n_classes = len(facies_labels)

    # Leave out one well for prediction
    well_names = df['WELL'].unique()
    f1 = []

    # Split data
    all_X = df.drop([TARGET_COLUMN, 'X_Axis', 'Y_Axis'], axis=1)
    all_Y = df[TARGET_COLUMN]

    all_output = []
    all_target = []

    for i in range(len(well_names)):
        if (test_well_names != 'all') and (test_well_names != well_names[i]):
            continue

        train_X = all_X[all_X['WELL'] != well_names[i]]
        train_Y = all_Y[all_X['WELL'] != well_names[i]]

        test_X = all_X[all_X['WELL'] == well_names[i]]
        test_Y = all_Y[all_X['WELL'] == well_names[i]]

        print(train_Y.value_counts())
        train_Y = train_Y.values

        train_X = train_X.drop(['WELL', 'DEPTH'], axis=1)
        test_X = test_X.drop(['WELL', 'DEPTH'], axis=1)

        X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2,
                                                              random_state=10*args.seed)

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_train = imp.fit_transform(X_train)
        X_valid = imp.transform(X_valid)
        test_X = imp.transform(test_X)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        test_X = scaler.transform(test_X)

        clf = TabNetClassifier(optimizer_params={'lr': 1e-3},
                               n_d=n_d, n_a=n_a,
                               n_steps=n_steps,
                               gamma=gamma,
                               scheduler_params={"step_size": 10, "gamma": 0.9},
                               scheduler_fn=torch.optim.lr_scheduler.StepLR,
                               mask_type='entmax'
                               )

        if use_pretrain:
            unsupervised_model = TabNetPretrainer(
                n_d=n_d, n_a=n_a,
                n_steps=n_steps,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=1e-3),
                mask_type='entmax' #'entmax' or 'sparsemax'
            )

            unsupervised_model.fit(
                X_train=np.vstack([X_train, test_X]),
                eval_set=[X_valid],
                pretraining_ratio=0.75,
                max_epochs=300,
            )

            # Model

            clf.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                max_epochs=200,
                from_unsupervised=unsupervised_model
            )
        else:
            clf.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                max_epochs=25
            )

        predictions = clf.predict(test_X)
        output_X['Tabnet Output'] = predictions
        output_X['Facies'] = test_Y.values

        all_target += test_Y.tolist()
        all_output += predictions.tolist()


        # Print model report
        print("\n------------------------------------------------------")
        print("Leaving out well " + well_names[i])
        # Confusion Matrix
        conf = confusion_matrix(test_Y, predictions, labels=np.arange(6))
        # Print Results
        print("\nModel Report")
        print("-Accuracy: %.6f" % (accuracy(conf)))
        print("-F1 Score: %.6f" % (f1_score(test_Y, predictions, labels=np.arange(6), average='weighted')))
        f1.append(f1_score(test_Y, predictions, labels=np.arange(6), average='weighted'))
        print("\nConfusion Matrix Results")

        display_cm(conf, facies_labels[:6], display_metrics=True, hide_zeros=True)

        output_file.write("\n------------------------------------------------------\n")
        output_file.write("Leaving out well \n" + well_names[i])
        # Print Results
        output_file.write("\nModel Report\n")
        output_file.write("-Accuracy: %.6f\n" % (accuracy(conf)))
        output_file.write("-F1 Score: %.6f\n" % (f1_score(test_Y, predictions, labels=np.arange(6), average='weighted')))
        # output_file.write("\nConfusion Matrix Results")

        # display_cm(conf, facies_labels[:6], display_metrics=True, hide_zeros=True)
        output_X.to_csv('%s/%s_output.csv' % (log_folder, test_well_names[i]))


    print("\n------------------------------------------------------")
    print("Final Results")
    print("-Average accuracy Score: %6f" % (accuracy_score(all_target, all_output)))
    print("-Average F1 Score: %6f" % (sum(f1) / (1.0 * len(f1))))

    output_file.write("\n------------------------------------------------------\n")
    output_file.write("Final Results\n")
    output_file.write("-Average F1 Score: %6f\n" % (sum(f1) / (1.0 * len(f1))))

    output_file.close()


if __name__ == "__main__":
    print("Run baseline")

    args = get_parser().parse_args()
    train_model(args)
