import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def gb_model(df, STATE_ID, n_to_keep = 10, drop_pct = 0.2, to_impute = True, scale = False , **kwargs):

    df = df[df.STATE == STATE_ID].reset_index(drop=True)
    df = df.drop(columns = ['OBJECTID', 'OBJECTID_1','STATE'], inplace=False)
    df = df.loc[df.DESIGNATED.notnull()]

    drop_count = drop_pct * df.shape[0]
    null_cols = df.columns[df.isnull().sum() != 0]
    many_null_cols = []
    for col in null_cols:
        count = df[col].isna().sum()
        if count > drop_count:
            many_null_cols.append(col)

    df_clean = df.drop(columns=many_null_cols, inplace=False)
    y = df_clean.pop('DESIGNATED')
    X = df_clean.select_dtypes(np.number)
    y.loc[y == False] = 'False'
    y.loc[y == True] = 'True'
    y_map = {'NotEligible': 0,
             'False': 1,
             'True': 2}
    y = y.map(y_map)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=2)

    imp = SimpleImputer(strategy='mean')
    ss = StandardScaler()
    if to_impute:
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)
    if scale:
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)

    clf = GradientBoostingClassifier( **kwargs )
    clf.fit(X_train, y_train)
    cr = classification_report(y_test, clf.predict(X_test), output_dict=True)
    feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
    return feat_importances.nlargest(n_to_keep), clf.score(X_test, y_test), cr['2']['f1-score']





