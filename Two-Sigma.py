''' 

Two Sigma Financial modeling Kaggle competition
@author: Jithin Pradeep
@email : jithinpr2@gmail.com
@website: www.jithinjp.in

'''
import kagglegym
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

'''pending #todo task:
# Provide comments/documnet the code for future reference.
# Rectify the warning messages:
    1 ../src/script.py:42: SettingWithCopyWarning: 
      A value is trying to be set on a copy of a slice from a DataFrame.
      Try using .loc[row_indexer,col_indexer] = value instead

      See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      train[column + '_nan_'] = pd.isnull(train[column])

'''
# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col = [c for c in observation.train.columns if c not in excl]

train = pd.read_hdf('../input/train.h5')
train = train[col]
train_median= train.median(axis=0)

train = observation.train[col]
n = train.isnull().sum(axis=1)

for column in train.columns:
    train[column + '_nan_'] = pd.isnull(train[column])
    train_median[column + '_nan_'] = 0
    
train = train.fillna(train_median)
train['znull'] = n
n = []

et_reg = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=20, verbose=1)
model1 = et_reg.fit(train, observation.train['y'])


low_y = -0.080
high_y = 0.080

above_cutoff = (observation.train.y > high_y)
below_cutoff = (observation.train.y < low_y)
with_cutoff = (~above_cutoff & ~below_cutoff)

model2 = LinearRegression(n_jobs=-1)

model2.fit(np.array(observation.train[col].fillna(train_median).loc[with_cutoff, 'technical_20'].values).reshape(-1,1), 
           observation.train.loc[with_cutoff, 'y'])

train = []

ymedian_dict = dict(observation.train.groupby(["id"])["y"].median())

while True:
    test1 = observation.features[col]
    n = test1.isnull().sum(axis=1)
    
    for column in test1.columns:
        test1[column + '_nan_'] = pd.isnull(test1[column])
        
    test1 = test1.fillna(train_median)
    test1['znull'] = n
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    test2 = np.array(observation.features[col].fillna(train_median)['technical_20'].values).reshape(-1,1)
    
    target['y'] = (model1.predict(test1
                               ).clip(low_y, high_y) * 0.6) + (model2.predict(test2).clip(low_y, high_y) * 0.3)
    
    target['y'] = target.apply(lambda r: 0.9 * r['y'] + 0.05 * ymedian_dict[r['id']] 
                           if r['id'] in ymedian_dict 
                           else r['y'], axis = 1)
    
    target['y'] = [float(format(x, '.6f')) for x in target['y']]
    

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print(reward)
    
    if done:
        print("el fin ...", info["public_score"])
        break
