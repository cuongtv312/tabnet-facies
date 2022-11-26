"""
Run the baseline from
    - https://github.com/seg/2016-ml-contest/blob/master/Stochastic_validations.ipynb
    - https://github.com/seg/2016-ml-contest/blob/master/LA_Team/Facies_classification_LA_TEAM_05_VALIDATION.ipynb
    - https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try03_v2_VALIDATION.ipynb


"""

import numpy as np
import pandas as pd
import os
import argparse
import xgboost as xgb
import sys
from models import Transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import torch

from sklearn.metrics import confusion_matrix, f1_score
from geotut_utils import accuracy, accuracy_adjacent, display_cm, adjacent_facies, get_parser,\
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

    base_folder = args.base_folder
    log_folder = utils.create_log_folder("./output", 'transformer', sys.argv)
    use_pretrain = args.use_pretrain
    use_seq_fts = args.use_seq_fts

    output_file = open('%s/log.txt' % log_folder, 'w')

    data, test_data = read_data(base_folder)
    if use_seq_fts:
        print('\t\t------------------')
        print('Add sequence features')
        seq_length = args.seq_length
        data, added_fts = add_seq_fts(data, seq_length, feature_names)
        test_data, _ =  add_seq_fts(test_data, seq_length, feature_names)

        use_features = feature_names + added_fts
    else:
        use_features = feature_names

    print("Feature length: ", len(use_features))

    # Leave out one well for prediction
    well_names = data['Well Name'].unique()
    f1 = []

    test_well_names = ['STUART', 'CRAWFORD']

    for i in range(len(test_well_names)):
        # Split data
        all_X = data.drop(['Facies', 'Formation', 'Depth'], axis=1)
        all_Y = data['Facies'] - 1

        train_X = all_X
        train_Y = all_Y
        test_X = test_data[test_data['Well Name'] == test_well_names[i]]
        test_Y = test_data[test_data['Well Name'] == test_well_names[i]]['Facies'].values - 1

        train_X = train_X[use_features].values
        test_X = test_X[use_features].values

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(train_X)

        train_X = imp.transform(train_X)
        test_X = imp.transform(test_X)

        # Model
        X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2,
                                                              random_state=10*args.seed)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        test_X = scaler.transform(test_X)

        # Group 5 rows as a sequence

        params = {'batch_size': args.batch_size,
          'shuffle': False,
          'drop_last': True, # Disregard last incomplete batch
          'num_workers': 2}

        params_test = {'batch_size': 1,
          'shuffle': False,
          'drop_last': False, # Disregard last incomplete batch
          'num_workers': 2}

        # Prepare data for RNN and LSTM model
        training_ds = WelllogDataset(X_train, y_train, args.seq_length)
        training_dl = DataLoader(training_ds, **params)

        valid_ds = WelllogDataset(X_valid, y_valid, args.seq_length)
        valid_dl = DataLoader(test_ds, **params_test)

        # Loss Function #
        criterion = torch.nn.CrossEntropyLoss()

        # Optimizer
        optim = torch.optim.Adam(model_final.parameters(), lr=args.lr, betas=(0.5, 0.999))

        model = Transformer(feature_size=X_train.shape[1],num_layers=1,dropout=0.1)

        # Train 
        print("Training {} using {} started with total epoch of {}.".format(model_final.__class__.__name__, "Adam", args.num_epochs))

        for epoch in range(args.num_epochs):
            for _, (X_train, label) in enumerate(training_dl):

                X_train = X_train.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)
                pred = model(X_train)
                label = label.squeeze()
                train_loss = criterion(pred, label)
                optim.zero_grad()
                train_loss.backward()
                optim.step()

                print(f"Epoch: {epoch}, Loss: {train_loss.item()}")

            # Test
            f1_test = []
            accuracy_test = []
            for _, (X_test, label) in enumerate(test_dl):
                X_test = X_test.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)

                model_final.eval()
                pred = model_final(X_test)
                pred = pred.argmax(dim=-1)
                correct = torch.sum(pred == label)

                f1.append(f1_score(label.detach().numpy(), pred.detach().numpy(), average="weighted"))
                f1_test.append(f1_score(label.detach().numpy(), pred.detach().numpy(), average="weighted"))
                accuracy_test.append(100 * correct / args.batch_size)

        # Test
        # Predict training set:
        predictions = model.predict(test_X)

        # Print model report
        print("\n------------------------------------------------------")
        print("Leaving out well " + test_well_names[i])
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
        output_file.write("-F1 Score: %.6f\n" % (f1_score(test_Y, predictions, labels=np.arange(6), average='weighted')))


    print("\n------------------------------------------------------")
    print("Final Results")
    print("-Average F1 Score: %6f" % (sum(f1) / (1.0 * len(f1))))

    output_file.write("\n------------------------------------------------------\n")
    output_file.write("Final Results\n")
    output_file.write("-Average F1 Score: %6f\n" % (sum(f1) / (1.0 * len(f1))))

    output_file.close()


if __name__ == "__main__":
    print("Run xgb baseline")

    train_model(get_parser(tabnet=True).parse_args())
