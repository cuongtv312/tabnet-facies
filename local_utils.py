import pandas as pd
from scipy.stats import mode
import argparse
import numpy as np

EPSILON = 1e-6


def calc_dif(x):
    x_diff = x[:-1] - x[:1]

    diff = np.mean(x_diff)

    return diff


# Apply an O(M+N) complexity algorithm for sampling the initial_values
def sampling_values_one_column(initial_depth, target_depth, initial_values,
                               aggregate_type, delta_range, add_stats_fts):
    i_start = 0
    i_end = 0

    target_values = np.zeros_like(target_depth)
    stats_fts = []
    for i in range(len(target_depth)):
        dleft = target_depth[i] - delta_range
        dright = target_depth[i] + delta_range

        while (i_start < len(initial_depth)):
            if initial_depth[i_start] >= dleft - EPSILON:
                break
            i_start += 1

        while (i_end < len(initial_depth) - 1):
            if initial_depth[i_end + 1] >= dright + EPSILON:
                break

            i_end += 1

        if i_end >= i_start:
            if aggregate_type == 'mean':
                value = np.mean(initial_values[i_start:i_end + 1])
                if add_stats_fts:
                    value_std = np.std(initial_values[i_start:i_end + 1])
                    value_max = np.max(initial_values[i_start:i_end + 1])
                    value_min = np.min(initial_values[i_start:i_end + 1])
                    value_range = value_max - value_min
                    value_dif = calc_dif(initial_values[i_start:i_end + 1])
                    stats_fts += [{'std': value_std,
                                   'max': value_max,
                                   'min': value_min,
                                   'range': value_range,
                                   'dif': value_dif}]
            else:
                value = mode(initial_values[i_start:i_end + 1])[0][0]
        else:
            # Got an empty range
            value = initial_values[i_start]

        target_values[i] = value

    if add_stats_fts:
        return target_values, pd.DataFrame(stats_fts)
    else:
        return target_values, None


def sampling_values(input_df, well_name, min_depth, max_depth, delta_d, aggregate_type_dict, add_stats_fts):
    depth = []
    start_d = min_depth
    while start_d <= max_depth + EPSILON:
        depth += [start_d]
        start_d += delta_d

    output_df = pd.DataFrame({
        'DEPTH': depth,
        'WELL': [well_name for _ in range(len(depth))]
    })

    new_fts = []
    for c in aggregate_type_dict:
        add_flag = add_stats_fts
        if aggregate_type_dict[c] != 'mean':
            add_flag = False

        new_values, stats_fts = sampling_values_one_column(initial_depth=input_df['DEPTH'].values,
                                                target_depth=depth,
                                                initial_values=input_df[c].values,
                                                aggregate_type=aggregate_type_dict[c],
                                                delta_range=delta_d/2,
                                                add_stats_fts=add_flag)
        output_df[c] = new_values

        if add_flag:
            for new_c in stats_fts.columns:
                output_df[c + '_' + new_c] = stats_fts[new_c]
                new_fts += [c + '_' + new_c]

    return output_df, new_fts


def get_facies_mapping(input_df, target_column):
    """
    The input df have several types of facies which are not continous values.
    This function mapping to 0 ... n_facies_classes-1, also give a label for each old values facies1, facies5

    :param input_df:
    :param target_column:
    :return:
    """
    list_values = list(np.unique(input_df[target_column].values))
    list_values = sorted(list_values)

    facies_labels = ['L%i' % i for i in list_values]
    m = dict([(list_values[i], i) for i in range(len(list_values))])
    print(list_values)

    input_df[target_column] = input_df[target_column].apply(lambda l: m[l])
    return input_df, facies_labels
