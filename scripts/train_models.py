import pandas as pd
import numpy as np
import os, json, itertools
from generate_model import *
from collections import Counter
import matplotlib.pyplot as plt
import us

#Load Premade dataframe that merges all 4 provided datasets
df_merged = pd.read_pickle('../data/AllDataMerged.pkl')

#The CONTIGUOUS variable is actually in the dropdown image so lets make it numerical
df_merged.CONTIGUOUS = 1*(df_merged.CONTIGUOUS.fillna(0))

with open('../data/variable_names.json', 'r') as fp:
    variable_translate = json.load(fp)
    #Define parameters for GradientBoostedClassfier

model_args = {'learning_rate':.7, 'n_estimators':75,
             'subsample':.8,'min_samples_split':2,
             'min_samples_leaf':1,'max_depth':4,
             'random_state':333,'max_features':'sqrt',
             'verbose':0,'max_leaf_nodes':None,
             'warm_start':False,'validation_fraction':0.1,
             'n_iter_no_change':10 }


# Generate a list of the 50 states + DC
state_list = [i for i in range(1,57)]
drop_list = [3,7,14,43,52]

for drop in drop_list:

    state_list.remove(drop)

# Model returns a list of n_to_keep most important features,
# classifier accuracy and True Designation f1-score
feat_dict = {}
scores = []
f1 = []
features =[]
state_names = []
for i in state_list:

    #a,b,c,d = gb_model(df_merged, i, n_to_keep=10 , **model_args)
    a,b,c,d = cleaned_data_model(df_merged, i, n_to_keep=10 , **model_args)

    feat_dict[i] = a
    features.append(a.index.to_list())
    scores.append(b)
    f1.append(c)
    state_names.append(us.states.lookup(str(i).zfill(2)).abbr)

# Flatten feature list then count what has the most

total_features = list(itertools.chain(*features))
feature_counts = Counter(total_features)
counts = pd.Series(feature_counts).sort_values(ascending=True)
counts.nlargest(15).plot(kind='barh')
plt.show()
for name in counts.nlargest(15).index:
    if name in variable_translate.keys():
        print(variable_translate[name])
    else :
        print(name)


