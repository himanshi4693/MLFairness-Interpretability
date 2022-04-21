import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score

#DATA RETRIEVAL
columns = ['age', 'working-class', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship-status', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income-cat']
census_income = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                        delimiter=',',header=None, names = columns, skipinitialspace = True)

#filter for only US citizens
census_income = census_income[census_income["native-country"] == 'United-States']

#replace nan with most freq val
census_income= census_income.replace(' ?', np.nan)
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
census_income_arr = np.array(imputer.fit_transform(census_income))
census_income_imp = pd.DataFrame(census_income_arr[:, :], columns= columns)

#drop columns
census_income_imp_filt = census_income_imp.drop(columns= ['education', 'native-country'])

#map target categories to 0 (unfavourable), 1(favourable)
census_income_imp_filt['income-cat'] = census_income_imp_filt['income-cat'].map({'<=50K': 0, '>50K': 1})

#saving as a csv, EDA visualization on notebook
census_income_imp_filt.to_csv('PreppedData/census_income.csv', index= False)


#DATA PREPARATION FOR MODEL TRAINING
census_income = pd.read_csv("PreppedData/census_income.csv", skipinitialspace = True)

num_features = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week', 'education-num', 'income-cat']
cat_features = ['occupation', 'working-class', 'marital-status', 'relationship-status', 'race', 'sex']

target = "income-cat"


# #one-hot encoding
dummies = pd.get_dummies(census_income_imp_filt[cat_features])
census_income_encoded = pd.concat([census_income_imp_filt[num_features], dummies], axis=1)

income_cat = census_income_encoded['income-cat']
census_income_encoded_X = census_income_encoded.drop(['income-cat'], axis=1)

income_X_train, income_X_test, income_Y_train, income_Y_test = train_test_split(census_income_encoded_X, income_cat, test_size=0.30, random_state=30)

#standardization
sc=StandardScaler()
income_X_train_std = sc.fit_transform(income_X_train)
income_X_test_std = sc.transform(income_X_test)
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
rf_random.fit(income_X_train_std, income_Y_train)
Y_test_pred = rf_random.predict(income_X_test_std)

income_test_outputs= pd.DataFrame()
income_test_outputs['GT_income'] = income_Y_test
income_test_outputs['predicted_income'] = Y_test_pred

#get original (non-encoded) test data
income_final= income_test_outputs.join(census_income_imp_filt)
income_final.to_pickle('PredsData/census_income_test.pkl')