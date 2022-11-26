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
import argparse
import xgboost as xgb
from models import CNN, LSTM, RNN, ResNetBaseline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from geotut_utils import accuracy, accuracy_adjacent, display_cm, adjacent_facies, get_parser
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
            y = self.y[idx + self.seq_length:idx + self.seq_length + 1].values
            # y = y - 1
            pairs.append((x, y))
        return pairs

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.metrics[idx]

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def make_sequecens(df, seq_length, features, target):
    data_length = len(df)
    pairs = []
    for idx in range(data_length - seq_length):
        x = df[idx : idx + seq_length][features].values
        y = df[idx + seq_length : idx + seq_length + 1][target].values
        pairs.append((x, y))
    return pairs


def train_model(args):
    utils.seed_everything(args.seed)

    base_folder = args.base_folder
    log_folder = utils.create_log_folder("./output", f"{args.model}_geotut", sys.argv)

    output_file = open('%s/log.txt' % log_folder, 'w')

    data, test_data = read_data(base_folder)

    # Leave out one well for prediction
    well_names = data['Well Name'].unique()
    f1 = []
    print(data['Facies'].unique())

    test_well_names = ['STUART', 'CRAWFORD']

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

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)
        
        # Group 5 rows as a sequence
        # print(type(train_X))
        # train_X = group_rows(5, train_X)
        # print(train_X)

        params = {'batch_size': args.batch_size,
          'shuffle': False,
          'drop_last': True, # Disregard last incomplete batch
          'num_workers': 2}

        params_test = {'batch_size': 1,
          'shuffle': False,
          'drop_last': False, # Disregard last incomplete batch
          'num_workers': 2}

        # Prepare data for RNN and LSTM model
        training_ds = WelllogDataset(train_X, train_Y, args.seq_length)
        training_dl = DataLoader(training_ds, **params)

        test_ds = WelllogDataset(test_X, test_Y, args.seq_length)
        test_dl = DataLoader(test_ds, **params_test)


        # train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_X.reshape(train_X.shape[0], 1,  train_X.shape[1])), torch.LongTensor(train_Y.values))
        # test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_X.reshape(test_X.shape[0], 1, test_X.shape[1])), torch.LongTensor(test_Y.values))
        # train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        # test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)

        # Model
        if args.model == 'cnn':
            model_final = CNN(args.seq_length, 40, args.num_classes, 3).to(device)
        elif args.model == 'lstm':
            model_final = LSTM(7, args.hidden_size, args.num_layers, args.num_classes, args.bidirectional).to(device)
        elif args.model == 'rnn':
            model_final = RNN(input_size=7, hidden_size=args.hidden_size, num_layers=args.num_layers, num_classes=args.num_classes).to(device)
        else:
            model_final = ResNetBaseline(in_channels=args.seq_length, num_pred_classes=args.num_classes).to(device)
        print(model_final)


        # Loss Function #
        criterion = torch.nn.CrossEntropyLoss()

        # Optimizer
        optim = torch.optim.Adam(model_final.parameters(), lr=args.lr, betas=(0.5, 0.999))
        
        # Train 
        print("Training {} using {} started with total epoch of {}.".format(model_final.__class__.__name__, "Adam", args.num_epochs))

        for epoch in range(args.num_epochs):
            for _, (X_train, label) in enumerate(training_dl):

                X_train = X_train.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.long)
                pred = model_final(X_train)
                label = label.squeeze()
                # print(label.shape, label.shape)
                # print(label)
                # print(pred.argmax(dim=1))
                train_loss = criterion(pred, label)

                # print(label, pred)
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

            f1.append(f1_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), average="weighted"))
            f1_test.append(f1_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), average="weighted"))
            accuracy_test.append(100 * correct / args.batch_size)
            

        output_file.write("\n------------------------------------------------------\n")
        output_file.write("Leaving out well \n" + well_names[i])
        # Print Results
        output_file.write("\nModel Report\n")
        output_file.write("-Accuracy: %.6f\n" % (sum(accuracy_test) / (1.0 * len(accuracy_test))))
        output_file.write("-F1 Score: %.6f\n" % (sum(f1_test) / (1.0 * len(f1_test))))



    print("\n------------------------------------------------------")
    print("Final Results")
    print("-Average F1 Score: %6f" % (sum(f1) / (1.0 * len(f1))))

    output_file.write("\n------------------------------------------------------\n")
    output_file.write("Final Results\n")
    output_file.write("-Average F1 Score: %6f\n" % (sum(f1) / (1.0 * len(f1))))

    output_file.close()


if __name__ == "__main__":
    args = get_parser(use_dnn=True).parse_args()
    print(f"Run {args.model} baseline")

    train_model(args)
