import numpy as np
import pandas as pd
import COMPASDataPrep
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import shap

recidivism_raw= COMPASDataPrep.compas_filtered.drop(['score_text', "decile_score"], axis=1) #to train a model without the COMPAS scores, and compare accuracy with model trained with

recidivism_num_features=["juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
recidivism_cat_features= ["sex", "age_cat", "race", "c_charge_degree", "two_year_recid"]

#one-hot encoding
dummies = pd.get_dummies(recidivism_raw[recidivism_cat_features])
recidivism_encoded= pd.concat([recidivism_raw[recidivism_num_features], dummies], axis=1)

recidivism = recidivism_encoded['two_year_recid']
recidivism_encoded_X = recidivism_encoded.drop(['two_year_recid'], axis=1)

recidivism_X_train, recidivism_X_test, recidivism_Y_train, recidivism_Y_test = train_test_split(recidivism_encoded_X, recidivism, test_size=0.30, random_state=30)

#standardization
sc=StandardScaler()
compas_X_train_std = sc.fit_transform(recidivism_X_train)
compas_X_test_std = sc.transform(recidivism_X_test)

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
rf_random.fit(compas_X_train_std, recidivism_Y_train)
Y_test_pred = rf_random.predict(compas_X_test_std)

recidivism_test_outputs= pd.DataFrame()
recidivism_test_outputs['GT_recidivism'] = recidivism_Y_test
recidivism_test_outputs['predicted_recidivism'] = Y_test_pred

#get original (non-encoded) test data
recidivism_final = recidivism_test_outputs.join(recidivism_raw)
recidivism_final.to_pickle('PredsData/recidivism_test.pkl')

