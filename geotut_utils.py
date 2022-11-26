# from https://raw.githubusercontent.com/seg/2016-ml-contest/master/utils.py
import numpy as np
import argparse
from torch.utils.data import Dataset
    
facies_colors = ['#F4D03F', '#F5B041', '#DC7633','#A569BD',
       '#000000', '#000080', '#2E86C1', '#AED6F1', '#196F3D']

adjacent_facies = np.array([[1], [0,2], [1], [4], [3,5], [4,6,7], [5,7], [5,6,8], [6,7]])

def calc_dif(x):
    x_diff = x[:-1] - x[:1]

    diff = np.mean(x_diff)

    return diff

def add_seq_fts(data, seq_length, feature_names):

    added_fts = []

    for ft_name in feature_names:
        values = data[ft_name].values

        value_std = [0 for _ in range(len(values))]
        value_max = [0 for _ in range(len(values))]
        value_min = [0 for _ in range(len(values))]
        value_range = [0 for _ in range(len(values))]
        value_dif =  [0 for _ in range(len(values))]
        

        for i in range(len(values)):
            i_start = i - seq_length // 2
            i_end = i + seq_length // 2

            if i_start < 0:
                i_start = 0
                i_end = seq_length-1
            elif i_end >= len(values) - 1:
                i_end = len(values) - 1
                i_start = i_end - (seq_length - 1)

            value_std[i] = np.std(values[i_start:i_end + 1])
            value_max[i] = np.max(values[i_start:i_end + 1])
            value_min[i] = np.min(values[i_start:i_end + 1])
            value_range[i] = value_max[i] - value_min[i]
            value_dif[i] = calc_dif(values[i_start:i_end + 1])

        data[ft_name + '_std'] = value_std
        data[ft_name + '_max'] = value_max
        data[ft_name + '_min'] = value_min
        data[ft_name + '_range'] = value_range
        data[ft_name + '_dif'] = value_dif

        added_fts += [ft_name + '_std', ft_name + '_max', ft_name + '_min', ft_name + '_range', ft_name + '_dif']

    return data, added_fts



def get_parser(use_dnn=False, tabnet=False, rtdl=False):
    parser = argparse.ArgumentParser()

    # General parameters
    # parser.add_argument('--base_folder', type=str, default='./input',
    #                     help='Base input folder')
    parser.add_argument('--test_well_names', type=str, default='test',
                        help='test or others')
    parser.add_argument('--use_seq_fts', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=2022,
                        help='Running seed')
    parser.add_argument('--base_folder', type=str, default='./input/geotut-contest-2016',
                        help='Base input folder')

    parser.add_argument('--seq_length', type=int, default=5, help='window size')

    if use_dnn:
        parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
        parser.add_argument('--model', type=str, default='xgb', choices=['cnn', 'rnn', 'lstm', 'resnet'])
        parser.add_argument('--input_size', type=int, default=1, help='input_size')
        parser.add_argument('--hidden_size', type=int, default=10, help='hidden_size')
        parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
        parser.add_argument('--output_size', type=int, default=1, help='output_size')
        parser.add_argument('--num_classes', type=int, default=9, help='num_classes')
        parser.add_argument('--bidirectional', type=bool, default=False, help='use bidirectional or not')

        parser.add_argument('--num_epochs', type=int, default=20, help='total epoch')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    elif tabnet:
        parser.add_argument('--use_pretrain', default=False, action='store_true')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--n_d', type=int, default=48, help='Width of the decision prediction layer')
        parser.add_argument('--n_a', type=int, default=48, help='Width of the attention embedding for each mask')
        parser.add_argument('--gamma', type=float, default=1.0,
                            help='This is the coefficient for feature reusage in the masks')
        parser.add_argument('--n_steps', type=int, default=5,
                            help='Number of steps in the architecture (usually between 3 and 10)')
    elif rtdl:
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    else:
        parser.add_argument('--model', type=str, default='xgb', choices=['xgb', 'rf', 'lightgbm'])

    return parser


def display_cm(cm, labels, hide_zeros=False, display_metrics=False):
    """Display confusion matrix with labels, along with
       metrics such as Recall, Precision and F1 score.
       Based on Zach Guo's print_cm gist at
       https://gist.github.com/zachguo/10296432
    """

    total_samples = cm.sum(axis = 1)
    test_mask = total_samples > 0

    precision = np.diagonal(cm)/cm.sum(axis=0).astype('float')
    recall = np.diagonal(cm)/cm.sum(axis=1).astype('float')
    F1 = 2 * (precision * recall) / (precision + recall)
    
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    F1[np.isnan(F1)] = 0
    
    total_precision = np.sum(precision * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    total_recall = np.sum(recall * cm.sum(axis=1)) / cm.sum(axis=(0,1))
    total_F1 = np.sum(F1[test_mask] * cm.sum(axis=1)[test_mask]) / cm.sum(axis=(0,1))
    #print total_precision
    
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + " Pred", end=' ')
    for label in labels: 
        print("%{0}s".format(columnwidth) % label, end=' ')
    print("%{0}s".format(columnwidth) % 'Total')
    print("    " + " True")
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=' ')
        for j in range(len(labels)): 
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeros:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            print(cell, end=' ')
        print("%{0}d".format(columnwidth) % sum(cm[i,:]))
        
    if display_metrics:
        print()
        print("Precision", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % precision[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_precision)
        print("   Recall", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % recall[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_recall)
        print("       F1", end=' ')
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % F1[j]
            print(cell, end=' ')
        print("%{0}.2f".format(columnwidth) % total_F1)
    
                  
def display_adj_cm(
        cm, labels, adjacent_facies, hide_zeros=False, display_metrics=False):
    """This function displays a confusion matrix that counts 
       adjacent facies as correct.
    """
    adj_cm = np.copy(cm)
    
    for i in np.arange(0,cm.shape[0]):
        for j in adjacent_facies[i]:
            adj_cm[i][i] += adj_cm[i][j]
            adj_cm[i][j] = 0.0
        
    display_cm(adj_cm, labels, hide_zeros, 
                             display_metrics)

def accuracy(conf):
    total_correct = 0.
    nb_classes = conf.shape[0]
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
    acc = total_correct/sum(sum(conf))
    return acc

def accuracy_adjacent(confusion_m, adjacent_facies):
    nb_classes = confusion_m.shape[0]
    total_correct = 0.
    for i in np.arange(0,nb_classes):
        total_correct += confusion_m[i][i]
        for j in adjacent_facies[i]:
            total_correct += confusion_m[i][j]
    return total_correct / sum(sum(confusion_m))

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
            pairs.append((x, y))
        return pairs

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.metrics[idx]

