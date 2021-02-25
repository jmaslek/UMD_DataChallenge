import pandas as pd
import os, json
from generate_model import gb_model
import matplotlib.pyplot as plt

#Load Premade dataframe that merges all 4 provided datasets
df_merged = pd.read_pickle('../data/AllDataMerged.pkl')

#Define parameters for GradientBoostedClassfier
model_args = {'learning_rate':.3, 'n_estimators':120,
             'subsample':.8,
              'min_samples_split':3,
             'min_samples_leaf':1,
             'max_depth':5,
              'random_state':40,
             'max_features':'sqrt',
              'verbose':0,
             'max_leaf_nodes':None,
              'warm_start':False,
             'validation_fraction':0.1,
              'n_iter_no_change':1 }


# Generate a list of the 50 states + DC
state_list = [i for i in range(1,57)]
drop_list = [3,7,14,43,52]

for drop in drop_list:
    state_list.remove(drop)

# Model returns a list of n_to_keep most important features,
# classifier accuracy and True Designation f1-score
feats = []
scores = []
f1 = []

for i in state_list:
    a,b,c = gb_model(df_merged, i, n_to_keep=10 , **model_args)
    feats.append(a)
    scores.append(b)
    f1.append(c)


