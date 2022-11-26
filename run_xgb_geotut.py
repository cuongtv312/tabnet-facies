"""
Run the baseline from
    - https://github.com/seg/2016-ml-contest/blob/master/Stochastic_validations.ipynb
    - https://github.com/seg/2016-ml-contest/blob/master/LA_Team/Facies_classification_LA_TEAM_05_VALIDATION.ipynb
    - https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try03_v2_VALIDATION.ipynb


"""

import sys
import numpy as np
import pandas as pd
import os
import xgboost as xgb
import lightgbm as lgbm

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from geotut_utils import accuracy, accuracy_adjacent, display_cm, adjacent_facies, get_parser, \
    add_seq_fts
import utils

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D', 'PS', 'BS']

feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']


def read_data(base_folder):

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
    print(df_train['Facies'].value_counts())
    print(test_data['Facies'].value_counts())
    return df_train, test_data


def get_accuracies(y_preds, df_test, predict_df):
    """
    Get the F1 scores from all the y_preds.
    y_blind is a 1D array. y_preds is a 2D array.
    """
    accs = []
    for y_pred in y_preds:
        predict_df['Facies'] = y_pred
        all_data = predict_df.join(df_test, how='inner')
        y_blind = all_data['LithCode'].values
        y_pred = all_data['Facies'].values
        y_pred = y_pred[y_blind != 11]
        y_blind = y_blind[y_blind != 11]
        cv_conf = confusion_matrix(y_blind, y_pred)
        accs.append(accuracy(cv_conf))

    return np.array(accs)


def train_model(args):
    utils.seed_everything(args.seed)
    model_type = args.model
    use_seq_fts = args.use_seq_fts

    base_folder = args.base_folder
    log_folder = utils.create_log_folder("./output", f"{model_type}_geotut", sys.argv)

    output_file = open('%s/log.txt' % log_folder, 'w')

    data, test_data = read_data(base_folder)

    # We based on seq length
    if use_seq_fts:
        print('Add sequence features')
        seq_length = args.seq_length
        data, _ = add_seq_fts(data, seq_length, feature_names)
        test_data, _ = add_seq_fts(test_data, seq_length)

    # Leave out one well for prediction
    well_names = data['Well Name'].unique()
    f1 = []

    test_well_names = ['STUART', 'CRAWFORD']

    all_output = []
    all_target = []

    for i in range(len(test_well_names)):
        # Split data
        all_X = data.drop(['Facies', 'Formation', 'Depth'], axis=1)
        all_Y = data['Facies'] - 1
        test_x = test_data.drop(['Facies', 'Formation', 'Depth'], axis=1)
        test_y = test_data['Facies'] - 1

        train_X = all_X[all_X['Well Name'] != well_names[i]]
        train_Y = all_Y[all_X['Well Name'] != well_names[i]]

        test_X = test_x[test_x['Well Name'] == test_well_names[i]]
        test_Y = test_y[test_x['Well Name'] == test_well_names[i]]

        train_X = train_X.drop(['Well Name'], axis=1).fillna(-1)
        test_X = test_X.drop(['Well Name'], axis=1).fillna(-1)

        # Model
        if model_type == 'rf':
            print("Train random forest classifier")
            clf = RandomForestClassifier(n_estimators=500, max_depth=6, max_samples=0.8, max_features=0.8) # RANDOM FORREST
            clf.fit(train_X, train_Y)
        elif model_type == 'lightgbm':
            clf = lgbm.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=6, learning_rate=0.1,
                                      n_estimators=500, subsample_for_bin=200000, objective=None, class_weight=None,
                                      min_split_gain=0.0, min_child_weight=0.001, min_child_samples=1, subsample=0.8,
                                      subsample_freq=0, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0,
                                      random_state=args.seed, n_jobs=-1, importance_type='split')
            clf.fit(train_X, train_Y)
        else:
            print("Train xgboost classifier")
            clf = xgb.XGBClassifier(learning_rate=0.01, n_estimators=500, max_depth=6,
                                            min_child_weight=1, gamma=0, subsample=0.8, reg_alpha=1,
                                            colsample_bytree=0.8, objective='multi:softmax',
                                            nthread=4, scale_pos_weight=1, seed=args.seed*10)
            # Fit the algorithm on the data
            clf.fit(train_X, train_Y)

            # Predict training set:
        predictions = clf.predict(test_X)
        test_X['%s_output' % model_type] = predictions
        
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
        print("-Adjacent Accuracy: %.6f" % (accuracy_adjacent(conf, adjacent_facies)))
        print("-F1 Score: %.6f" % (f1_score(test_Y, predictions, labels=np.arange(9), average='weighted')))
        f1.append(f1_score(test_Y, predictions, labels=np.arange(9), average='weighted'))
        print("\nConfusion Matrix Results")

        display_cm(conf, facies_labels, display_metrics=True, hide_zeros=True)
        output_file.write("\n------------------------------------------------------\n")
        output_file.write("Leaving out well \n" + well_names[i])
        # Print Results
        output_file.write("\nModel Report\n")
        output_file.write("-Accuracy: %.6f\n" % (accuracy(conf)))
        output_file.write("-F1 Score: %.6f\n" % (f1_score(test_Y, predictions, labels=np.arange(9), average='weighted')))

        test_data.to_csv('%s/%s_output.csv' % (log_folder, test_well_names[i]), )

    print("\n------------------------------------------------------")
    print("Final Results")
    print("-Average F1 Score: %6f" % (sum(f1) / (1.0 * len(f1))))
    print("-Average accuracy Score: %6f" % (accuracy_score(all_target, all_output)))

    output_file.write("\n------------------------------------------------------\n")
    output_file.write("Final Results\n")
    output_file.write("-Average F1 Score: %6f\n" % (sum(f1) / (1.0 * len(f1))))

    output_file.close()


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(f"Run {args.model} baseline")

    train_model(args)
