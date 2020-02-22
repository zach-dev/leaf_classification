import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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

# cross validation svm run
svm_cv_results = []
for num_folds in range(2, 11):
    # create svm object
    svm_clf = SVC(kernel='rbf', probability=True)

    # perform grid search cross validation runs
    parameters = {'C': [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3,
                        2**-2, 2**-1, 2**1, 2**2, 2**3],
                  'gamma': [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3,
                            2**-2, 2**-1, 2**1, 2**2, 2**3]}
    svm_cv = GridSearchCV(svm_clf, param_grid=parameters, cv=num_folds, n_jobs=8, return_train_score=True)
    svm_cv.fit(x_train, y_train)

    # preserve results of run
    print(num_folds, max(svm_cv.cv_results_['mean_test_score']))
    print(svm_cv.best_estimator_)

    svm_cv_results.append([num_folds, np.max(svm_cv.cv_results_['mean_train_score']), np.max(svm_cv.cv_results_['mean_test_score']),
                           np.mean(svm_cv.cv_results_['mean_score_time']), svm_cv.best_estimator_])


# create a data frame with the final data
svm_cv_results = pd.DataFrame(svm_cv_results)
svm_cv_results.columns = ['num_folds', 'cv_train_score', 'cv_test_score', 'time', 'best_est']

# plot test error for each of the Ks for the 10 fold run
plt.plot(svm_cv_results.loc[:, 'num_folds'], svm_cv_results.loc[:, 'cv_test_score'])
plt.xlabel('number of folds')
plt.ylabel('best test set prediction accuracy')
plt.title("Test Set Prediction Accuracy vs K for Face Classification")
plt.show()

# tighten the range over which C is checked and re-run grid search with only 9 fold cross validation
# perform grid search cross validation runs
parameters = {'C': [2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7,
                    7.25, 7.5, 7.75]}
svm_cv_ref = GridSearchCV(svm_clf, param_grid=parameters, cv=5, n_jobs=8, return_train_score=True)
svm_cv_ref.fit(x_train, y_train)

# fit best model to full training data set and do prediction of test set
# typically probably wouldn't do this, but because the training data set is on the small side, it may be helpful
t1 = time.time()
best_svm_clf = svm_cv_ref.best_estimator_
best_svm_clf.fit(x_train, y_train)
t = time.time() - t1
svm_test_pred = best_svm_clf.predict_proba(x_test)

# format for submission
sub = pd.DataFrame(svm_test_pred, columns=list(label_enc.classes_))
sub.insert(0, 'id', test_ids)
sub.reset_index()
sub.to_csv('svm_submission.csv', index=False)

