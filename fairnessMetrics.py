import numpy as np

#RAW DATASET METRICS (EDA)
import pandas as pd


def base_rate(P, N):
    return round(P/(P+N), 2)

def accuracy(TP, TN, P, N):
    return round((TP + TN)/(P + N), 2)

def selection_rate(PP, PN): #rate of positive predictions (wrt total predictions)
    return round(PP/(PP+PN), 2)

def positive_predictive_value(TP, FP):
    return round(TP/(TP+FP), 2)

def true_pos_rate(TP, P):
    return round(TP/P, 2)

def true_neg_rate(TN, N):
    return round(TN/N, 2)

def false_pos_rate(FP, N):
    return round(FP/N, 2)

def false_neg_rate(FN, P):
    return round(FN/P, 2)

# def GT_disparate_impact():
#     #could be defined not just based on the outcome,
#     #but also some of the features that are relevant for prediction?
#     #for instance, in the adult dataset, the distribution education could differ across genders and races
#     return
#
# def GT_demographic_parity_difference():
#     #find pairwise diff of all base rates and create a df
#     return
#
# def pred_disparate_impact(): #this has to be defined wrt (predicted) outcomes,
#     # and can only be compared to the component of GT_disparate_impact which
#     # is calculated wrt true outcomes
#     return
#
# def false_positives_difference():
#     return
#
# def false_negatives_difference():
#     return
#
# def error_rate_difference():
#     return
#
# def error_rate_ratio():
#     return
#
# def pred_demographic_parity_difference():
#     return
#
# def pred_accuracy():
#     return
#
# def average_odds_difference():
#     return
#
# def equal_opportunity_difference():
#     return



