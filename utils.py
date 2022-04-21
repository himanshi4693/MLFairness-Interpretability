import numpy as np
import pandas as pd
import itertools as it
import utils, fairnessMetrics, Dataset


# given a list of protected attributes, return all possible combinations as a single list of conditions
def create_conditions(X_prot, prot_feature_names):
    prot_feature_names.sort() #since dataset columns are also sorted alphabetically
    dict_unique_values = {feature: [] for feature in prot_feature_names}
    for prot_feature in prot_feature_names:
        ix = prot_feature_names.index(prot_feature)
        if len(prot_feature_names) == 1:
            unique_values_list = list(np.unique(X_prot))
        else:
            unique_values_list = list(np.unique(X_prot[:, ix]))
        dict_unique_values[prot_feature] = unique_values_list
    combinations = it.product(*(dict_unique_values[prot_feature] for prot_feature in prot_feature_names))
    # a list of dicts, each dict representing a single combination
    conditions = [{prot_feature_names[k]: j for k, j in zip(range(len(prot_feature_names)), i)} for i in combinations]
    return conditions


def compute_boolean_conditioning_vector(X_prot, prot_feature_names, condition=None):
    # condition expects a list of dictionaries.

    # each dictionary can have one or more conditions as key-value pairs, with each key-value pair
    # representing the protected attribute and value (privileged or unprivileged)
    # to condition on. every condition within the dict must be satisfied for the group of conditions
    # to evaluate to true, therefore an AND operator is used in the inner for loop.
    # returns a vector of booleans, evaluating to True where atleast one of the dicts within the list
    # evaluates to true, therefore an OR operator is used in the outer loop.

    if condition is None:
        return np.ones(X_prot.shape[0], dtype=bool)

    cumulative_condition_vec = np.zeros(X_prot.shape[0], dtype=bool)
    for group in condition:
        group_condition_vec = np.ones(X_prot.shape[0], dtype=bool)
        for key, value in group.items():
            ix = prot_feature_names.index(key)
            group_condition_vec = np.logical_and(group_condition_vec, X_prot[:, ix] == value)
        cumulative_condition_vec = np.logical_or(cumulative_condition_vec, group_condition_vec)

    return cumulative_condition_vec


def compute_T_N(y_true, favourable_outcome, X_prot, X_prot_feature_names, condition=None):
    condition_vector = compute_boolean_conditioning_vector(X_prot, X_prot_feature_names, condition)

    y_actual_positive = np.logical_and(y_true == favourable_outcome, condition_vector)
    y_actual_negative = np.logical_and(y_true != favourable_outcome, condition_vector)

    return dict(
        P=np.sum(y_actual_positive),
        N=np.sum(y_actual_negative)
    )


def compute_TP_FN(y_true, y_predicted, favourable_outcome, X_prot, X_prot_feature_names, condition=None):
    # condition on protected attributes (optional)
    condition_vector = compute_boolean_conditioning_vector(X_prot, X_prot_feature_names, condition)

    y_actual_positive = np.logical_and(y_true == favourable_outcome, condition_vector)
    y_actual_negative = np.logical_and(y_true != favourable_outcome, condition_vector)
    y_predicted_positive = np.logical_and(y_predicted == favourable_outcome, condition_vector)
    y_predicted_negative = np.logical_and(y_predicted != favourable_outcome, condition_vector)

    return dict(
        P=np.sum(y_actual_positive),
        N=np.sum(y_actual_negative),
        PP=np.sum(y_predicted_positive),
        PN=np.sum(y_predicted_negative),
        TP=np.sum([np.logical_and(y_actual_positive, y_predicted_positive)]),
        FP=np.sum([np.logical_and(y_actual_negative, y_predicted_positive)]),
        TN=np.sum([np.logical_and(y_actual_negative, y_predicted_negative)]),
        FN=np.sum([np.logical_and(y_actual_positive, y_predicted_negative)]) #a false negative is a negative prediction which is incorrectly predicted
    )

def getMetrics(dataset, attributes_to_protect):

    X_prot, protected_features_names, Y_true, Y_pred, favourable_outcome = Dataset.getDataset(dataset, attributes_to_protect)

    conditions = utils.create_conditions(X_prot, protected_features_names)

    for condition in conditions:
        T_N_dict = utils.compute_T_N(Y_true, favourable_outcome, X_prot, protected_features_names, [condition])
        TP_FN_dict = utils.compute_TP_FN(Y_true, Y_pred, favourable_outcome, X_prot, protected_features_names, [condition])
        condition['base_success_rate'] = fairnessMetrics.base_rate(T_N_dict['P'], T_N_dict['N'])
        condition['accuracy'] = fairnessMetrics.accuracy(TP_FN_dict['TP'], TP_FN_dict['TN'], TP_FN_dict['P'], TP_FN_dict['N'])
        condition['error_rate'] = round(1 - condition['accuracy'], 2)
        condition['selection_rate']  = fairnessMetrics.selection_rate(TP_FN_dict['PP'], TP_FN_dict['PN'])
        condition['positive_predictive_value']  = fairnessMetrics.positive_predictive_value(TP_FN_dict['TP'], TP_FN_dict['FP']) #precision
        condition['TPR'] = fairnessMetrics.true_pos_rate(TP_FN_dict['TP'], TP_FN_dict['P']) #recall, or sensitivity
        condition['FPR'] = fairnessMetrics.false_pos_rate(TP_FN_dict['FP'], TP_FN_dict['N'])
        condition['TNR'] = fairnessMetrics.true_neg_rate(TP_FN_dict['TN'], TP_FN_dict['N']) #specificity
        condition['FNR'] = fairnessMetrics.false_neg_rate(TP_FN_dict['FN'], TP_FN_dict['P'])

    df_metrics = pd.DataFrame(conditions)

    return(df_metrics)

def getFairnessMetrics(metrics, attributes_to_protect):
    if len(attributes_to_protect) > 1:
        metrics['group'] = ''
        for attr in attributes_to_protect:
            metrics['group'] = metrics['group'] + ',' + metrics[attr]

        metrics['group'] = metrics['group'].str.lstrip(',')
        metrics = metrics.loc[:, ~metrics.columns.isin(attributes_to_protect)]
        protected_attr = "group"
    elif len(attributes_to_protect) == 1:
        protected_attr = attributes_to_protect[0]

    # get combinations
    metrics = metrics.set_index(protected_attr)
    cc = list(it.combinations(metrics.index, 2))  # list of tuples

    pairs_metrics = pd.DataFrame([])

    for c in cc:
        df = metrics.loc[c, :]
        df[protected_attr] = df.index
        df = df.reset_index(drop=True)
        df1 = df.iloc[[0]].add_suffix('_1')
        df1['tmp'] = 1
        df2 = df.iloc[[1]].add_suffix('_2')
        df2['tmp'] = 1
        df_concat = pd.merge(df1, df2, on=['tmp']).drop('tmp', axis=1)
        pairs_metrics = pairs_metrics.append(df_concat)

    # reduce to interesting pairs
    if len(attributes_to_protect) > 1:
        pairs_metrics['group_1'] = pairs_metrics['group_1'].str.split(',')
        pairs_metrics['group_2'] = pairs_metrics['group_2'].str.split(',')
        # if attr to prot > 1, identify the protected variable value along which any difference metric can be computed
        pairs_metrics['common'] = pairs_metrics.apply(lambda x: list(set(x['group_1']).intersection(set(x['group_2']))),
                                                      axis=1)
        # print(pairs_metrics)
        #reduce to only rows with something in common, since these are the interesting pairs to compare
        pairs_metrics = pairs_metrics[pairs_metrics['common'].map(lambda c: len(c) > 0)]

    # to do: create functions for each fairness metric
    # call the functions for the entire df and compute metrics as separate columns
    pairs_metrics['disparate_impact'] = pairs_metrics.apply(lambda x: round(x['selection_rate_1'] / x['selection_rate_2'], 2) if(x['selection_rate_2'] !=0) else None, axis=1)
    pairs_metrics['demographic_parity_diff']= pairs_metrics.apply(lambda x: round(x['selection_rate_1']-x['selection_rate_2'], 2), axis=1)
    pairs_metrics['equal_opportunity_difference']= pairs_metrics.apply(lambda x: round(x['TPR_1']-x['TPR_2'], 2), axis=1) #TPR difference
    pairs_metrics['false_positive_rate_difference']= pairs_metrics.apply(lambda x: round(x['FPR_1']-x['FPR_2'], 2), axis=1)
    pairs_metrics['false_negative_rate_difference']= pairs_metrics.apply(lambda x: round(x['FNR_1']-x['FNR_2'], 2), axis=1)
    pairs_metrics['true_negative_rate_difference']= pairs_metrics.apply(lambda x: round(x['TNR_1']-x['TNR_2'], 2), axis=1)
    pairs_metrics['average_odds_difference']= pairs_metrics.apply(lambda x: round(0.5 * (x['false_positive_rate_difference'] + x['equal_opportunity_difference']), 2), axis=1)
    pairs_metrics['average_abs_odds_difference']= pairs_metrics.apply(lambda x: round(0.5 * (np.abs(x['false_positive_rate_difference']) + np.abs(x['equal_opportunity_difference'])), 2), axis=1)
    pairs_metrics['error_rate_difference']= pairs_metrics.apply(lambda x: round(0.5 * (x['error_rate_1'] - x['error_rate_2']), 2), axis=1)
    pairs_metrics['error_rate_ratio']= pairs_metrics.apply(lambda x: round(0.5 * (x['error_rate_1'] / x['error_rate_2']), 2) if(x['error_rate_2'] !=0) else None, axis=1)
    return(pairs_metrics)