import numpy as np
import pandas as pd
import COMPASDataPrep

def getDataset(dataset_name, attrs_to_protect):
        if dataset_name == 'compas':
            dataset = COMPASDataPrep.compas_filtered
            Y_true = dataset['two_year_recid'].to_numpy() # true labels
            Y_pred = np.where(dataset['decile_score'] < 5, 0, 1) # risk predictions from COMPAS: decile score < 5 is
            # considered low, hence favourable i.e. value 0, same as in recidivism
            favourable_outcome = 0

        if dataset_name == 'recidivism':
            dataset = pd.read_pickle('PredsData/recidivism_test.pkl')
            Y_true = dataset['GT_recidivism'].to_numpy()  # true labels
            Y_pred = dataset['predicted_recidivism']
            favourable_outcome = 0  #non repeating

        elif dataset_name == 'german':
            dataset= pd.read_pickle('PredsData/german_credit_test.pkl')
            Y_true = dataset['GT_risk'].to_numpy()  # true labels
            Y_pred = dataset['predicted_risk']
            dataset['age'] = np.where(dataset['age'] <= 25, '<=25', '>25')
            favourable_outcome = 1 #good credit

        elif dataset_name == 'income':
            dataset= pd.read_pickle('PredsData/census_income_test.pkl')
            Y_true = dataset['GT_income'].to_numpy()  # true labels
            Y_pred = dataset['predicted_income']
            favourable_outcome = 1 #>50k income

        protected_features_names = attrs_to_protect
        X_prot = dataset.loc[:, dataset.columns.isin(protected_features_names)].to_numpy()
        # possible_conditions = utils.create_conditions(X_prot, protected_features_names)  # making conditions based on the given protected attributes
        return X_prot, protected_features_names, Y_true, Y_pred, favourable_outcome