from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import statistics

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)))

def cross_val(model,X,y):
    kfold = KFold(10, True, 1)
    # enumerate splits
    MPE = []
    for train, test in kfold.split(X):
        y_train = y[train]; X_train = X[train,:]
        y_test = y[test]; X_test = X[test,:]
        mod = model
        mod.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        MPE.append(mean_absolute_percentage_error(y_test, y_pred))
    return MPE

# Data preparation
df = read_csv('/Users/mengkaixu/Desktop/Dataset/energydata_complete.csv')
drop_col = ['Appliances', 'date', 'rv1', 'rv2']
X = df.drop(drop_col, axis=1).values
y = df.values[:,1]
# Feature selection based on importance
model1 = RandomForestRegressor()
# fit the model
model1.fit(X, y)
# get importance
importance_R = model1.feature_importances_
plt.bar([x for x in range(len(importance_R))], importance_R)

''' Random Forest '''
score = []
# Feature selection based on Recursive feature selection
for i in range(4,15,1):
    a = i+1
    selector = RFE(model1,a, step=1)
    selector = selector.fit(X, y)
    important = selector.support_
    ranking = selector.ranking_
    res = sorted(range(len(ranking)), key=lambda sub: ranking[sub])[:a]
    XX = X[:,res]
    # random forest regressor
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    # cross_validation score
    RF_MPE = cross_val(model,XX,y)
    score.append(statistics.mean(RF_MPE))

# save score
np.save('/Users/mengkaixu/PycharmProjects/Mengkai1/score.npy', score)
# load score
ss = np.load('/Users/mengkaixu/PycharmProjects/Mengkai1/score.npy', allow_pickle=True)

''' Gradient Boosting '''
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

model2 = GradientBoostingRegressor()
# fit the model on the whole dataset
model2.fit(X, y)
# get importance
plt.figure()
importance_G = model2.feature_importances_
plt.bar([x for x in range(len(importance_G))], importance_G)
# Cross-validation
score_GB = []
# Feature selection based on Recursive feature selection
for i in range(0,25,1):
    a = i+1
    selector = RFE(model2, a, step=1)
    selector = selector.fit(X, y)
    important = selector.support_
    ranking = selector.ranking_
    res = sorted(range(len(ranking)), key=lambda sub: ranking[sub])[:a]
    XX = X[:, res]
    # random forest regressor
    model = GradientBoostingRegressor(n_estimators = 200, learning_rate = 0.1, loss = 'ls')
    # cross_validation score
    RF_MPE = cross_val(model,XX,y)
    score_GB.append(statistics.mean(RF_MPE))


''' LSTM '''
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional

score_LSTM = []
# Feature selection based on Recursive feature selection
for i in range(0,25,1):
    a = i+1
    selector = RFE(model1,a, step=1)
    selector = selector.fit(X, y)
    important = selector.support_
    ranking = selector.ranking_
    res = sorted(range(len(ranking)), key=lambda sub: ranking[sub])[:a]
    XX = X[:,res]
    # LSTM
    X1 = XX.reshape((XX.shape[0], 1, XX.shape[1]))
    model3 = Sequential()
    model3.add(Bidirectional(LSTM(200, activation='relu'), input_shape=(1, XX.shape[1])))
    model3.add(Dense(1))
    model3.compile(optimizer='adam', loss='mse')
    # cross_validation score
    RF_MPE = cross_val(model3,X1,y)
    score_LSTM.append(statistics.mean(RF_MPE))




