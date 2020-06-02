from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import statistics
from multiprocessing import Pool

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)))

def cross_val(model):
    df = read_csv('/Users/mengkaixu/Desktop/Dataset/energydata_complete.csv')
    drop_col = ['Appliances', 'date', 'rv1', 'rv2']
    X = df.drop(drop_col, axis=1).values
    y = df.values[:, 1]
    kfold = KFold(10, True, 1)
    # enumerate splits
    MPE = []
    for train, test in kfold.split(X):
        y_train = y[train];
        X_train = X[train, :]
        y_test = y[test];
        X_test = X[test, :]
        mod = model
        mod.fit(X_train, y_train)
        y_pred = mod.predict(X_test)
        MPE.append(mean_absolute_percentage_error(y_test, y_pred))
    return MPE

regressors_list = [RandomForestRegressor()]

if __name__ == '__main__':
    p = Pool(processes=20)
    LL = p.map(cross_val, regressors_list) # with multiple arguments
    p.close()
    p.join()
    print(LL)
