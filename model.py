import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xg

dataset = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

X_train = dataset.iloc[:,1:42].values
Y_train = dataset.iloc[:,42].values
X_test = test.iloc[:,1:42].values


np.corrcoef(X_train[:,37],Y_train[:])

"""
#Training XGBoost model
x= xg.DMatrix(X_train, label = Y_train)
X_test = xg.DMatrix(X_test)
param = {'eta':0.15,'max_depth':10,'objective':'multi:softmax','metric':'multi_logloss',
         'num_class':3,'num_leaves':512,'boosting':'dart'}
classifier = xg.train(param,x)
y_pred = classifier.predict(X_test)
"""

#Grid Search, hyperparameter optimization
param_test = {'learning_rate':[0.2,0.6,0.8],'max_depth':[7,8,10]}
grid_search = GridSearchCV(estimator =xg.XGBClassifier(learning_rate=0.1,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax' ),
                           param_grid = param_test,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
model = grid_search.fit(X_train, Y_train)
y_pred = model.predict(X_test)

#Best parameter and score from GridSearch
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print(best_accuracy,best_parameters)

#Saving the results in .csv
df = pd.DataFrame(y_pred[:],columns = ['Target'])
col1 = pd.read_csv('sample_submission.csv')
col1 = col1.drop('target',axis=1)
sub = pd.DataFrame({'connection_id':col1['connection_id'],'target':df['Target']})
sub.to_csv('output2.csv', index=False)
