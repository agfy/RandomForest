import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import cross_val_score

data = pd.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = data[data.columns[0:8]].as_matrix()
y = data['Rings'].as_matrix()

kf = KFold(n_splits=5, random_state=1, shuffle=True)

for n in range(1, 51):

    clf = RandomForestRegressor(n_estimators=n)
    score_arr = cross_val_score(clf, X, y, cv=kf, scoring='r2')

    print n, np.mean(score_arr)
