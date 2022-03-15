import pickle
import pandas as pd

from typing import Union

features = ['age', 'num_unpaid_bills',
            'merchant_category', 'merchant_group',
            'max_paid_inv_0_12m', 'avg_payment_span_0_12m',
            'status_max_archived_0_24_months', 'num_arch_ok_0_12m']

cat_fields = ['merchant_category', 'merchant_group']

def update_pipeline(X: Union[str, pd.DataFrame], y=None):
    """
    Function updating the Pipeline in case new data is available. Ideally one
    would reconsider the whole modelling framework if enough data becomes
    available.
    
    params:
        - X: str or pd.DataFrame with the file to the data/features or the data
        - y: default features

    """
    if isinstance(X, str):
        _X = pd.read_csv(X, delimiter=';')
        return update_pipeline(_X)

    if 'default' in X.columns:
        return update_pipeline(X.drop(['default'], X['default']))

    missing_set = set(features).difference(set(X.columns.to_list()))

    if missing_set:
        raise ValueError(f'The following features are missing: {missing_set}')

    X = X.loc[~(y.isna())].reset_index(drop=True)
    
    ohe = pickle.load(open(r'../2. Scripts/pickles/ohe.pkl', 'rb'))
    X_ohe = pd.DataFrame(enc.fit_transform(X[cat_fields]), 
                        columns=enc.get_feature_names_out()).reset_index()
    X = pd.concat([X[features].drop(columns=cat_fields, axis=1), X_ohe], axis=1)
    
    pipe = pickle.load(open(r'../2. Scripts/pickles/pipeline.pkl', 'rb'))
    res=pipe.fit(X, y)

    pickle.dump(res, open(r'../2. Scripts/pickles/pipeline.pkl', 'wb'))
    

def default_predictor(X: Union[str, pd.DataFrame], outfile=None):
    """
    Function producing a Default probability per 'uuid' given an available
    pipeline and available features
    
    params:
        - X: str or pd.DataFrame with the file to the data/features or the data
        - outfile: Optional, output dir of the prediction

    return:
        - pd.DataFrame with two column 'uuid' and 'default'
    """
  
    if isinstance(X, str):
        _X = pd.read_csv(X, delimiter=';')
        return default_predictor(_X)
    
    X = X.reset_index(drop=True)

    missing_set = set(features).difference(set(X.columns.to_list()))

    if missing_set:
        raise ValueError(f'The following features are missing: {missing_set}')

    ohe = pickle.load(open(r'../2. Scripts/pickles/ohe.pkl', 'rb'))
    pipe = pickle.load(open(r'../2. Scripts/pickles/pipeline.pkl', 'rb'))

    X_ohe = pd.DataFrame(ohe.transform(X[cat_fields]),
                         columns=ohe.get_feature_names_out()).reset_index(drop=True)

    X_prob = pd.concat([X[features].drop(columns=cat_fields, axis=1), X_ohe],
                        axis=1)
    
    default_probability = pipe.predict_proba(X_prob)[::, -1]
    
    df = pd.DataFrame({
                         'uuid': X['uuid'].values,
                         'default': default_probability
                         })
    if outfile:
        df.to_csv(outfile, index=False)

    return df
    
