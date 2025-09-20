import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from tqdm.auto import tqdm 

class NumericPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features):
        self.numeric_features = numeric_features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.numeric_features].copy()
        return X.values
    
    def get_feature_names(self):
        return self.numeric_features

class BoolPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, bool_features):
        self.bool_features = bool_features
        self.ordinary_encoders = None

    def fit(self, X, y=None):
        
        self.ordinary_encoders = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value = np.nan)
        self.ordinary_encoders.fit(X[self.bool_features])
        return self

    def transform(self, X, y=None):
        Xout = self.ordinary_encoders.transform(X[self.bool_features]).astype("float32")
        return Xout 

    def get_feature_names(self):
        return self.bool_features


class OneHotPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, onehot_features):
        self._feature_names = []
        self.onehot_features = onehot_features
        self.onehot_encoders = None
        self.labels = None

    def fit(self, X, y=None):
        
        self.onehot_encoders = {f:OneHotEncoder(handle_unknown='ignore',sparse_output=False) for f in self.onehot_features}
        self.labels = {}

        for c in self.onehot_features:
            self.onehot_encoders[c].fit(self._fillna(X[c]))
            self.labels[c] = self.onehot_encoders[c].categories_
            
        # Update feature names
        self._feature_names = []
        for k, v in self.labels.items():
            for f in k + '_' + v[0]:
                self._feature_names.append(f)
                
        return self

    def transform(self, X, y=None):
        Xs = [self.onehot_encoders[c].transform(self._fillna(X[c])) for c in self.onehot_features]
        return np.hstack(Xs)

    def get_feature_names(self):
        return self._feature_names
    
    def _fillna(self, col):
        return col.astype(str).values.reshape(-1, 1)

class EmbeddingPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, embedding_features, max_dictionary_size=200):
        self.embedding_features = embedding_features
        self.max_dictionary_size = max_dictionary_size 
        self.trans_dics = {}

    def fit(self, X, y=None):
        self.trans_max = {}
        for col in tqdm(self.embedding_features):
            Xcol = self._fillna(X[col])
            self.trans_dics[col] = self._get_trans_dic(Xcol)
            self.trans_max[col] = max(list(self.trans_dics[col].values()))
        
        return self

    def transform(self, X, y=None):
        Xs = [self._fillna(X[col]).apply(
            lambda a:self.trans_dics[col].get(a,self.trans_max[col])) 
              for col in self.embedding_features]
        return pd.concat(Xs,axis=1).values 
    
    def _fillna(self, seri):
        return seri.astype(str)
    
    def _get_trans_dic(self,seri):
        max_size = self.max_dictionary_size
        dfi = pd.DataFrame(seri.value_counts()).rename(lambda _:"count",axis = 1)
        dfi.sort_values("count",ascending=False,inplace = True)
        
        n = len(dfi)
        if n<=max_size:
            dfi["index"] = range(n)
        else:
            dfi["index"] = [i if i<=max_size-1 else max_size-1 for i in range(n)]

        trans_dic = dict(dfi["index"])  
        return trans_dic
    
    def get_feature_names(self):
        return self.embedding_features

class TabularPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 bool_features = None,
                 onehot_features = None,
                 embedding_features = None,
                 numeric_features = None,
                 cat_features= None,
                 onehot_max_cat_num = 9,
                 embedding_max_dictionary_size = 1000, 
                 normalization = "quantile",
                 **kwargs):
        super().__init__(**kwargs)
        
        self.bool_features = bool_features
        self.onehot_features = onehot_features
        self.embedding_features = embedding_features
        self.numeric_features = numeric_features
        self.cat_features = cat_features
        self.onehot_max_cat_num = onehot_max_cat_num
        self.embedding_max_dictionary_size = embedding_max_dictionary_size 
        self.normalization = normalization
        self.pipeline = None
        self.feature_names = None
        self.embedding_cats_num = {}
        self.is_fitted = False
        
    def fit(self, X:pd.DataFrame, y=None) -> pd.DataFrame:
        
        if self.cat_features is None:
            self.cat_features = X.select_dtypes(
                include=["category","object","int8","int32","int64"]).columns.tolist()
            
        dic_cat_nums = {col:len(X[col].value_counts()) for col in self.cat_features}
        
        if self.bool_features is None:
            self.bool_features = [col for col in self.cat_features 
                                  if dic_cat_nums[col]==2]
        if self.onehot_features is None:
            self.onehot_features = [col for col in self.cat_features 
                                    if 3<= dic_cat_nums[col] <= self.onehot_max_cat_num]
        if self.embedding_features is None:
            self.embedding_features = [col for col in self.cat_features 
                                       if dic_cat_nums[col]> self.onehot_max_cat_num]
        if self.numeric_features is None:
            self.numeric_features = [col for col in X.columns if col not in
            self.bool_features+self.onehot_features+self.embedding_features]
            
        transformer_list = []
        
        if self.numeric_features:
            
            if self.normalization == 'quantile':
                normalizer = QuantileTransformer(output_distribution='normal')
            elif self.normalization == 'standard':
                normalizer = StandardScaler()
            elif self.normalization == 'minmax':
                normalizer = MinMaxScaler()
            else:
                raise Exception("normalization type should be in ('quantile','standard','minmax')!")
            
            numeric_pipeline = Pipeline(steps=[
                ('preparator', NumericPreprocessor(numeric_features=self.numeric_features)),
                ('imputer', SimpleImputer(strategy="mean")),
                (self.normalization, normalizer)
            ])
            transformer_list.append(('numeric', numeric_pipeline))
            
        if self.bool_features:
            bool_pipeline = Pipeline(steps=[
                ("preparator", BoolPreprocessor(bool_features=self.bool_features)),

            ])
            transformer_list.append(('bool',bool_pipeline))

        if self.onehot_features:
            onehot_pipeline = Pipeline(steps=[
                ("preparator", OneHotPreprocessor(onehot_features=self.onehot_features)),

            ])
            transformer_list.append(('onehot', onehot_pipeline))

        if self.embedding_features:
            embedding_pipeline = Pipeline(steps=[
                ("preparator", EmbeddingPreprocessor(
                    embedding_features=self.embedding_features,
                    max_dictionary_size=self.embedding_max_dictionary_size)),

            ])
            transformer_list.append(('embedding', embedding_pipeline))
            max_dic_size = self.embedding_max_dictionary_size
            self.embedding_cats_num = {col:min(dic_cat_nums[col],max_dic_size) for col in self.embedding_features}

        self.pipeline = FeatureUnion(transformer_list=transformer_list)
        self.pipeline.fit(X)
        
        self.feature_names = []
        for pipe in transformer_list:
            self.feature_names.extend(pipe[-1][0].get_feature_names())
            
        self.is_fitted = True
            
        return self

    def transform(self, X: pd.DataFrame, y = None) -> pd.DataFrame:
        Xout = self.pipeline.transform(X)
        dfout = pd.DataFrame(Xout,columns = self.feature_names,index = X.index)
        for col in dfout.columns:
            if col in self.embedding_features:
                dfout[col] = dfout[col].astype("int32")
            else:
                dfout[col] = dfout[col].astype("float32")
        return dfout
    
    def get_feature_names(self):
        return self.feature_names
    
    def get_embedding_features(self):
        return self.embedding_features
    
    def get_numeric_features(self):
        embedding = set(self.embedding_features)
        return [col for col in self.feature_names if col not in embedding]
    
    def get_embedding_cats_num(self):
        return self.embedding_cats_num