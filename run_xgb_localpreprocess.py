"""
Run the baseline with default features


"""

import numpy as np
import pandas as pd
import os
import argparse
import xgboost as xgb
import sys
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgbm

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from geotut_utils import accuracy, display_cm
from local_utils import get_facies_mapping
import utils

TARGET_COLUMN = 'Facies_Full PETREL'
DELTA_DEPTH = 0.5

# Should get len(TRAIN_COLUMNS) + 3 columns: WELL, DEPTH, TARGET_COLUMN
TRAIN_COLUMNS = ['CAL', 'GR', 'PHI', 'LLD', 'RHOB',
                 'LLS', 'PEF-4PE', 'Perm', 'PHIE', 'PHIT']
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
    parser.add_argument('--seed', type=int, default=2022,
                        help='Running seed')
    parser.add_argument('--use_seq_fts', default=False, action='store_true')

    parser.add_argument('--model', type=str, default='xgb', choices=['xgb', 'rf', 'lightgbm'])

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

def train_model(args):
    utils.seed_everything(args.seed)
    model_type = args.model

    base_folder = args.base_folder
    test_well_names = args.test_well_names

    use_seq_fts = args.use_seq_fts

    log_folder = utils.create_log_folder("./output", 'xgb_localdata', sys.argv)

    output_file = open('%s/log.txt' % log_folder, 'w')

    data = read_data(base_folder, add_stats_fts=args.use_seq_fts)
    print(data.shape)

    df, facies_labels = get_facies_mapping(data, TARGET_COLUMN)
    print(df[TARGET_COLUMN].unique())
    print(facies_labels)
    n_classes = len(facies_labels)

    # Leave out one well for prediction
    well_names = df['WELL'].unique()
    f1 = []

    all_X = df.drop(['DEPTH', TARGET_COLUMN], axis=1)
    all_Y = df[TARGET_COLUMN]

    all_output = []
    all_target = []


    for i in range(len(well_names)):
        if (test_well_names != 'all') and (test_well_names != well_names[i]):
            continue


        # Split data
        train_X = all_X[all_X['WELL'] != well_names[i]]
        train_Y = all_Y[all_X['WELL'] != well_names[i]]

        test_X = all_X[all_X['WELL'] == well_names[i]]
        test_Y = all_Y[all_X['WELL'] == well_names[i]]

        print('Train labels')
        print(train_Y.value_counts())
        print(train_X.shape)

        print('Test labels')
        print(test_Y.value_counts())
        print(test_X.shape)

        train_X = train_X.drop(['WELL', 'X_Axis', 'Y_Axis'], axis=1).fillna(-1e3)
        test_X = test_X.drop(['WELL', 'X_Axis', 'Y_Axis'], axis=1).fillna(-1e3)

        # Model
        if model_type == 'rf':
            print("Train random forest classifier")
            clf = RandomForestClassifier(n_estimators=500, max_depth=6, max_samples=0.8, max_features=0.8) # RANDOM FORREST
            clf.fit(train_X, train_Y)
        elif model_type == 'lightgbm':
            clf = lgbm.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=6, learning_rate=0.1,
                                      n_estimators=500, subsample_for_bin=200000, objective=None, class_weight=None,
                                      min_split_gain=0.0, min_child_weight=0.001, min_child_samples=1, subsample=0.2,
                                      subsample_freq=0, colsample_bytree=0.2, reg_alpha=0.0, reg_lambda=0.0,
                                      random_state=args.seed, n_jobs=-1, importance_type='split')
            clf.fit(train_X, train_Y)
        else:
            clf = xgb.XGBClassifier(learning_rate=0.01, n_estimators=500, max_depth=3,
                                        min_child_weight=1, gamma=0, subsample=0.8, reg_alpha=1.,
                                        colsample_bytree=0.5, objective='multi:softmax',
                                        nthread=8, seed=27)

        # Fit the algorithm on the data
            clf.fit(train_X, train_Y, eval_metric='merror')

        # model_final = RandomForestClassifier(n_estimators=1000) # RANDOM FORREST
        # model_final.fit(train_X, train_Y)

        # Predict training set:
        predictions = clf.predict(test_X)

        all_target += test_Y.tolist()
        all_output += predictions.tolist()

        # Print model report
        print("\n------------------------------------------------------")
        print("Leaving out well " + well_names[i])
        # Confusion Matrix
        conf = confusion_matrix(test_Y, predictions, labels=np.arange(9))
        # Print Results
        print("\nModel Report")
        print("-Accuracy: %.6f" % (accuracy(conf)))
        print("-F1 Score: %.6f" % (f1_score(test_Y, predictions, labels=np.arange(9), average='weighted')))
        f1.append(f1_score(test_Y, predictions, labels=np.arange(9), average='weighted'))
        print("\nConfusion Matrix Results")

        display_cm(conf, facies_labels[:9], display_metrics=True, hide_zeros=True)

        output_file.write("\n------------------------------------------------------\n")
        output_file.write("Leaving out well \n" + well_names[i])
        # Print Results
        output_file.write("\nModel Report\n")
        output_file.write("-Accuracy: %.6f\n" % (accuracy(conf)))
        output_file.write("-F1 Score: %.6f\n" % (f1_score(test_Y, predictions, labels=np.arange(9), average='weighted')))
        # output_file.write("\nConfusion Matrix Results")

        # display_cm(conf, facies_labels[:6], display_metrics=True, hide_zeros=True)


    print("\n------------------------------------------------------")
    print("Final Results")
    print("-Average accuracy Score: %6f" % (accuracy_score(all_target, all_output)))
    print("-Average F1 Score: %6f" % (sum(f1) / (1.0 * len(f1))))

    output_file.write("\n------------------------------------------------------\n")
    output_file.write("Final Results\n")
    output_file.write("-Average F1 Score: %6f\n" % (sum(f1) / (1.0 * len(f1))))

    output_file.close()


if __name__ == "__main__":
    print("Run xgb baseline")

    args = get_parser().parse_args()
    train_model(args)
