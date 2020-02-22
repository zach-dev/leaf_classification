import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score

'''
Perform classification of leaves using Kth nearest neighbor technique.
'''

# load test and train data
train_data = pd.read_csv('../../data/train.csv')
test_data  = pd.read_csv('../../data/test.csv')

# break out predictors and create test and train x variables
x_train_dev = train_data.loc[:, 'margin1':'texture64']
x_test = test_data.loc[:, 'margin1':'texture64']
test_ids = test_data.loc[:, 'id']

# encode leaf labels and create test and train y variables
label_enc = LabelEncoder()
label_enc.fit(train_data.species)
y_train_dev = label_enc.transform(train_data.species)

# cross validation knn run
knn_cv_results = []
for num_folds in range(2, 20):
    for K in range(1, 16):
        # create knn object
        knn_clf = KNeighborsRegressor(n_neighbors=K)

        # perform k-fold cross validation run with specified number of folds and K
        knn_cv = cross_validate(knn_clf, x_train_dev, y_train_dev, cv=num_folds, n_jobs=-1, return_train_score=True)

        # preserve results of run
        knn_cv_results.append([K, num_folds, np.mean(knn_cv['train_score']), np.mean(knn_cv['test_score']),
                               np.mean(knn_cv['score_time'])])

# create a data frame with the final data
knn_cv_results = pd.DataFrame(knn_cv_results)
knn_cv_results.columns = ['K', 'num_folds', 'cv_train_score', 'cv_test_score', 'time']

# plot test error for each of the Ks for the 10 fold run
a = knn_cv_results.loc[knn_cv_results['num_folds'] == 10]
plt.plot(a.loc[:, 'K'], a.loc[:, 'cv_test_score'])
plt.xlabel('K')
plt.ylabel('test set prediction accuracy')
plt.title("Test Set Prediction Accuracy vs K for Leaf Classification with 10 CV folds")
plt.show()

# fit best model to full training data set and do prediction of test set
best_knn_clf = KNeighborsRegressor(n_neighbors=1)
best_knn_clf.fit(x_train_dev, y_train_dev)
knn_test_pred = best_knn_clf.predict_proba(x_test)

# format for submission
sub = pd.DataFrame(knn_test_pred, columns=list(label_enc.classes_))
sub.insert(0, 'id', test_ids)
sub.reset_index()
sub.to_csv('knn_submission.csv', index=False)

