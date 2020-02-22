import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
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

# pca
pca = PCA()
pca.fit(x_train)
x_train_pca = pca.transform(x_train)

# lda with singular value decomposition solver
lda_clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None)
lda_cv = cross_validate(lda_clf, x_train_pca, y_train, cv=10, n_jobs=1, return_train_score=True)

# fit best model to full training data set and do prediction of test set
# typically probably wouldn't do this, but because the training data set is on the small side, it may be helpful
lda_ws_clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None)
lda_ws_clf.fit(x_train_pca, y_train)
lda_test_pred = lda_ws_clf.predict_proba(x_test)

# format for submission
sub = pd.DataFrame(lda_test_pred, columns=list(label_enc.classes_))
sub.insert(0, 'id', test_ids)
sub.reset_index()
sub.to_csv('pca_lda_submission.csv', index=False)

# lda transform of x_train_pca
lda_clf.fit(x_train_pca, y_train)
x_train_pca_lda = lda_clf.transform(x_train_pca)

# svc run with x_train_pca_lda
# create svc object
svc_clf = SVC(kernel='linear', probability=True)

# perform grid search cross validation runs
parameters = {'C': [2 ** -15, 2 ** -14, 2 ** -13, 2 ** -12, 2 ** -11, 2 ** -10, 2 ** -9, 2 ** -8, 2 ** -7, 2 ** -6,
                    2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 1, 2 ** 2, 2 ** 3]}
svc_cv = GridSearchCV(svc_clf, param_grid=parameters, cv=10, n_jobs=8, return_train_score=True)
svc_cv.fit(x_train_pca_lda, y_train)

# tighten the range over which C is checked and re-run grid search with 10 fold cv
parameters = {'C':[0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016,
                   0.0017, 0.0018, 0.0019]}
svc_cv_ref = GridSearchCV(svc_clf, param_grid=parameters, cv=10, n_jobs=8, return_train_score=True, scoring='accuracy')
svc_cv_ref.fit(x_train_pca_lda, y_train)

# fit best model to full training data set and do prediction of test set
# typically probably wouldn't do this, but because the training data set is on the small side, it may be helpful
best_svc_clf = svc_cv_ref.best_estimator_
best_svc_clf.fit(x_train, y_train)
svc_test_pred = best_svc_clf.predict_proba(x_test)

# format for submission
sub = pd.DataFrame(svc_test_pred, columns=list(label_enc.classes_))
sub.insert(0, 'id', test_ids)
sub.reset_index()
sub.to_csv('pca_lda_svc_acc_submission.csv', index=False)

# repeat using log loss as scoring metric
# tighten the range over which C is checked and re-run grid search with 10 fold cv
parameters = {'C':[0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016,
                   0.0017, 0.0018, 0.0019]}
svc_cv_ref = GridSearchCV(svc_clf, param_grid=parameters, cv=10, n_jobs=8, return_train_score=True, scoring='log_loss')
svc_cv_ref.fit(x_train_pca_lda, y_train)

# fit best model to full training data set and do prediction of test set
# typically probably wouldn't do this, but because the training data set is on the small side, it may be helpful
best_svc_clf = svc_cv_ref.best_estimator_
best_svc_clf.fit(x_train, y_train)
svc_test_pred = best_svc_clf.predict_proba(x_test)

# format for submission
sub = pd.DataFrame(svc_test_pred, columns=list(label_enc.classes_))
sub.insert(0, 'id', test_ids)
sub.reset_index()
sub.to_csv('pca_lda_svc_ll_submission.csv', index=False)