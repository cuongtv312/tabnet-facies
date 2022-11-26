"""
Run the baseline with default features


"""

import numpy as np
import pandas as pd
import os
import argparse
import xgboost as xgb
import sys

from sklearn.metrics import confusion_matrix, f1_score
from geotut_utils import accuracy, display_cm
from local_utils import sampling_values, get_facies_mapping
import utils

TARGET_COLUMN = 'Facies_Full PETREL'
DELTA_DEPTH = 0.5

# Should get len(TRAIN_COLUMNS) + 3 columns: WELL, DEPTH, TARGET_COLUMN
TRAIN_COLUMNS = ['CAL', 'GR-3,5 ECGR', 'NPHI-3 TNPH', 'LLD-2,3RT-5RT_HRLT', 'RHOB',
                 'LLS-2,3,5 RLA1', 'PEF-4PE', 'Perm', 'PHIE', 'PHIT',
                 'SW', 'SWT', 'VCL_2VSH']
#TRAIN_COLUMNS = ['CAL', 'GR-3,5 ECGR', 'NPHI-3 TNPH', 'RHOB',
#                 'Perm', 'PHIE', 'PHIT',
#                 'SW', 'SWT']


# Read and preprocessing data
def read_data(base_folder, add_stats_fts=False):

    print('Read input data')
    filename = os.path.join(base_folder, 'preprocess_full_5wells.csv')
    df_train = pd.read_csv(filename)
    df_train['WELL'] = df_train['WELL'].astype('category')

    print('Sampling the target with a fixed distance')
    aggregate_type = dict([(c, 'mean') for c in TRAIN_COLUMNS])
    aggregate_type[TARGET_COLUMN] = 'count'

    well_names = df_train['WELL'].unique()
    list_dfs = []
    list_added_fts = None

    for c in well_names:
        sub_df = df_train[df_train['WELL'] == c].copy().reset_index(drop=True)
        min_depth = int(np.min(sub_df['DEPTH'].values)) + 100
        max_depth = int(np.max(sub_df['DEPTH'].values)) - 100
        print("Process well = ", c)
        print('\tmin depth = ', min_depth)
        print('\tmax depth = ', max_depth)

        tmp_df, tmp_fts = sampling_values(sub_df, c, min_depth, max_depth,
                                          delta_d=0.5, aggregate_type_dict=aggregate_type,
                                          add_stats_fts=add_stats_fts)
        print('\t', tmp_df.shape)
        list_dfs += [tmp_df]

        if add_stats_fts and list_added_fts is None:
            list_added_fts = tmp_fts

    return pd.concat(list_dfs, axis=0), list_added_fts


def train_model(args):
    utils.seed_everything(args.seed)

    base_folder = args.base_folder
    test_well_names = args.test_well_names

    log_folder = utils.create_log_folder("./output", 'xgb_localdata', sys.argv)

    output_file = open('%s/log.txt' % log_folder, 'w')

    data, list_new_fts = read_data(base_folder, add_stats_fts=args.add_stats_fts)
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

    for i in range(len(well_names)):
        if (test_well_names != 'all') and (test_well_names != well_names[i]):
            continue

        # Split data
        train_X = all_X[all_X['WELL'] != well_names[i]]
        train_Y = all_Y[all_X['WELL'] != well_names[i]]

        test_X = all_X[all_X['WELL'] == well_names[i]]
        test_Y = all_Y[all_X['WELL'] == well_names[i]]

        print(train_Y.value_counts())

        train_X = train_X.drop(['WELL'], axis=1)
        test_X = test_X.drop(['WELL'], axis=1)

        # Model
        model_final = xgb.XGBClassifier(learning_rate=0.01, n_estimators=500, max_depth=3,
                                        min_child_weight=20, gamma=0, subsample=0.75, reg_alpha=1.,
                                        colsample_bytree=0.25, objective='multi:softmax',
                                        nthread=8, seed=27)

        # Fit the algorithm on the data
        model_final.fit(train_X, train_Y, eval_metric='merror')

        # model_final = RandomForestClassifier(n_estimators=1000) # RANDOM FORREST
        # model_final.fit(train_X, train_Y)

        # Predict training set:
        predictions = model_final.predict(test_X)

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


    print("\n------------------------------------------------------")
    print("Final Results")
    print("-Average F1 Score: %6f" % (sum(f1) / (1.0 * len(f1))))

    output_file.write("\n------------------------------------------------------\n")
    output_file.write("Final Results\n")
    output_file.write("-Average F1 Score: %6f\n" % (sum(f1) / (1.0 * len(f1))))

    output_file.close()


if __name__ == "__main__":
    print("Run xgb baseline")

    args = get_parser().parse_args()
    train_model(args)
