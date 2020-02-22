import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

'''
Perform classification of leaves using SVC
'''

# load test and train data
train_data = pd.read_csv('../../data/train.csv')
test_data  = pd.read_csv('../../data/test.csv')

# break out predictors and create test and train x variables
x_train = train_data.loc[:, 'margin1':'texture64']
x_test = test_data.loc[:, 'margin1':'texture64']
test_ids = test_data.loc[:, 'id']

# encode leaf labels and create test and train y variables
label_enc = LabelEncoder()
label_enc.fit(train_data.species)
y_train = label_enc.transform(train_data.species)

# use stadardScaler to standardize data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# set up data splits for cv
sss = StratifiedShuffleSplit(n_splits=10, random_state=42)
sss.get_n_splits(x_train, y_train)

# initialize cv run storage variables
qda_cv_train_score = []
qda_cv_test_score = []

# vary covariance regularization parameter
for rp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    qda_train_score = []
    qda_test_score = []

    # perform cross validation
    for trn_idx, dev_idx in sss.split(x_train, y_train):
        # configure the train and dev data
        x_trn = x_train[trn_idx]
        x_dev = x_train[dev_idx]
        y_trn = y_train[trn_idx]
        y_dev = y_train[dev_idx]

        # qda
        qda_clf = QuadraticDiscriminantAnalysis(reg_param=rp)
        qda_clf.fit(x_trn, y_trn)
        qda_train_score.append(qda_clf.score(x_trn, y_trn))
        qda_test_score.append(qda_clf.score(x_dev, y_dev))

    qda_cv_train_score.append(np.mean(qda_train_score))
    qda_cv_test_score.append(np.mean(qda_test_score))

print(qda_cv_train_score, qda_cv_test_score)

# fit best model to full training data set and do prediction of test set
# typically probably wouldn't do this, but because the training data set is on the small side, it may be helpful
t1 = time.time()
qda_best_clf = QuadraticDiscriminantAnalysis(reg_param=0.4)
qda_best_clf.fit(x_train, y_train)
t = time.time() - t1
qda_test_pred = qda_best_clf.predict_proba(x_test)

# format for submission
sub = pd.DataFrame(qda_test_pred, columns=list(label_enc.classes_))
sub.insert(0, 'id', test_ids)
sub.reset_index()
sub.to_csv('qda_submission.csv', index=False)

