import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
lda_svd_train_score = []
lda_svd_test_score = []
lda_eig_train_score = []
lda_eig_test_score = []
lda_eig_ws_train_score = []
lda_eig_ws_test_score = []
lda_ls_train_score = []
lda_ls_test_score = []
lda_ls_ws_train_score = []
lda_ls_ws_test_score = []

# perform cross validation
for trn_idx, dev_idx in sss.split(x_train, y_train):
    # configure the train and dev data
    x_trn = x_train[trn_idx]
    x_dev = x_train[dev_idx]
    y_trn = y_train[trn_idx]
    y_dev = y_train[dev_idx]

    # lda with singular value decomposition solver
    lda_svd_clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None)
    lda_svd_clf.fit(x_trn, y_trn)
    lda_svd_train_score.append(lda_svd_clf.score(x_trn, y_trn))
    lda_svd_test_score.append(lda_svd_clf.score(x_dev, y_dev))

    # lda with eigenvalue decomposition solver
    lda_eig_clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=None)
    lda_eig_clf.fit(x_trn, y_trn)
    lda_eig_train_score.append(lda_eig_clf.score(x_trn, y_trn))
    lda_eig_test_score.append(lda_eig_clf.score(x_dev, y_dev))

    # lda with eigenvalue decomposition solver
    lda_eig_ws_clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    lda_eig_ws_clf.fit(x_trn, y_trn)
    lda_eig_ws_train_score.append(lda_eig_ws_clf.score(x_trn, y_trn))
    lda_eig_ws_test_score.append(lda_eig_ws_clf.score(x_dev, y_dev))

    # lda with least squares solver
    lda_ls_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
    lda_ls_clf.fit(x_trn, y_trn)
    lda_ls_train_score.append(lda_ls_clf.score(x_trn, y_trn))
    lda_ls_test_score.append(lda_ls_clf.score(x_dev, y_dev))

    # lda with least squares solver
    lda_ls_ws_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda_ls_ws_clf.fit(x_trn, y_trn)
    lda_ls_ws_train_score.append(lda_ls_ws_clf.score(x_trn, y_trn))
    lda_ls_ws_test_score.append(lda_ls_ws_clf.score(x_dev, y_dev))

lda_svd_cv_train_score = np.mean(lda_svd_train_score)
lda_svd_cv_test_score = np.mean(lda_svd_test_score)
lda_eig_cv_train_score = np.mean(lda_eig_train_score)
lda_eig_cv_test_score = np.mean(lda_eig_test_score)
lda_eig_ws_cv_train_score = np.mean(lda_eig_ws_train_score)
lda_eig_ws_cv_test_score = np.mean(lda_eig_ws_test_score)
lda_ls_cv_train_score = np.mean(lda_ls_train_score)
lda_ls_cv_test_score = np.mean(lda_ls_test_score)
lda_ls_ws_cv_train_score = np.mean(lda_ls_ws_train_score)
lda_ls_ws_cv_test_score = np.mean(lda_ls_ws_test_score)

print(lda_svd_cv_train_score, lda_svd_cv_test_score)
print(lda_eig_cv_train_score, lda_eig_cv_test_score)
print(lda_eig_ws_cv_train_score, lda_eig_ws_cv_test_score)
print(lda_ls_cv_train_score, lda_ls_cv_test_score)
print(lda_ls_ws_cv_train_score, lda_ls_ws_cv_test_score)

# fit best model to full training data set and do prediction of test set
# typically probably wouldn't do this, but because the training data set is on the small side, it may be helpful
t1 = time.time()
lda_eig_ws_clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
lda_eig_ws_clf.fit(x_train, y_train)
t = time.time() - t1
lda_test_pred = lda_eig_ws_clf.predict_proba(x_test)

t1 = time.time()
lda_eig_ws_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
lda_eig_ws_clf.fit(x_train, y_train)
t = time.time() - t1
lda_test_pred = lda_eig_ws_clf.predict_proba(x_test)

# format for submission
sub = pd.DataFrame(lda_test_pred, columns=list(label_enc.classes_))
sub.insert(0, 'id', test_ids)
sub.reset_index()
sub.to_csv('lda_submission.csv', index=False)

