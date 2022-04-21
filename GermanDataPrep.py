import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score

#DATA RETRIEVAL

column_names = ['status', 'month', 'credit_history',
            'purpose', 'credit_amount', 'savings', 'employment',
            'investment_as_income_percentage', 'personal_status',
            'other_debtors', 'residence_since', 'property', 'age',
            'installment_plans', 'housing', 'number_of_credits',
            'skill_level', 'people_liable_for', 'telephone',
            'foreign_worker', 'credit']

german_df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                        delimiter=' ',header=None, names= column_names)

# DATA TRANSFORMATION, as per the attributes information given on https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
#adding 'sex' column
status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
german_df['sex'] = german_df['personal_status'].replace(status_map)

german_df= german_df.drop('personal_status', axis=1)

#map bad credit(value 2) -> 0, good credit (value 1) -> 1
german_df['credit'] = german_df['credit'].map({1: 1, 2: 0})

#data imputation for savings account and checking account status

german_df['savings'] = german_df['savings'].fillna('A65')
german_df['status'] = german_df['status'].fillna('A14')

#saving as a csv, EDA visualization on notebook
german_df.to_csv('PreppedData/german_credit.csv', index= False)


#DATA PREPARATION FOR MODEL TRAINING
german_credit = pd.read_csv('PreppedData/german_credit.csv')

german_cat_features = ['status', 'credit_history', 'purpose',
                     'savings', 'employment', 'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level', 'telephone',
                     'foreign_worker', 'sex']

german_num_features=['age', 'month', 'credit_amount', 'investment_as_income_percentage','residence_since', 'number_of_credits', 'people_liable_for', 'credit']

#one-hot encoding
dummies = pd.get_dummies(german_credit[german_cat_features])
german_credit_encoded= pd.concat([german_credit[german_num_features], dummies], axis=1)

credit_risk = german_credit_encoded['credit']
german_credit_encoded_X = german_credit_encoded.drop(['credit'], axis=1)

german_X_train, german_X_test, german_Y_train, german_Y_test = train_test_split(german_credit_encoded_X, credit_risk, test_size=0.30, random_state=30)

#standardization
sc=StandardScaler()
german_X_train_std = sc.fit_transform(german_X_train)
german_X_test_std = sc.transform(german_X_test)
#
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]# Number of trees in random forest
max_features = ['auto', 'sqrt']# Number of features to consider at every split
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]# Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4]# Minimum number of samples required at each leaf node
bootstrap = [True, False]# Method of selecting samples for training each tree

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

random_forest = RandomForestClassifier(random_state = 100)
rf_random = RandomizedSearchCV(estimator = random_forest, param_distributions = random_grid, n_iter = 50, cv = 5, verbose=4, scoring='recall', random_state=42, n_jobs = -1)
rf_random.fit(german_X_train_std, german_Y_train)
Y_test_pred = rf_random.predict(german_X_test_std)

german_test_outputs= pd.DataFrame()
german_test_outputs['GT_risk'] = german_Y_test
german_test_outputs['predicted_risk'] = Y_test_pred

#get original (non-encoded) test data
german_final= german_test_outputs.join(german_credit)
german_final.to_pickle('PredsData/german_credit_test.pkl')

