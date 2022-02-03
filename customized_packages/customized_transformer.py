import numpy as np
import pandas as pd
from functools import reduce

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer



class NumericalAttributesTransformer(BaseEstimator, TransformerMixin):
    """
    Transform a numercial attributes before training
    
    Parameters
    ----------
    numerical_attributes : list
        list of numerical attributes which need to transform
    to_binary_attributes: list or str
        attributes to transform to binary
    to_log_attributes: list or str
        attributes to transform by log
    log_transform : bool, default=True
        If True, apply log1p to transform the data

    """
    
    def __init__(self, numerical_attributes, to_binary_attributes, to_log_attributes, log_transform=True):
        self.numerical_attributes = numerical_attributes
        self.to_binary_attributes = to_binary_attributes
        self.to_log_attributes = to_log_attributes
        self.log_transform = log_transform

    def fit(self, X, y=None):
        # do nothing
        return self

    def transform(self, X, y=None):
        
        if self.to_binary_attributes in self.numerical_attributes:
            X[self.to_binary_attributes + "_binary"] = np.where(X[self.to_binary_attributes] == 999, 1, 0)
            
            # drop pdays column
            X.drop(self.to_binary_attributes, axis = 1, inplace = True)
            
        if self.log_transform:
            # Transform log features
            for column in self.to_log_attributes:
                X[column] = np.log1p(X[column])
            
            # Take back column names
            self.columns_list = X.columns.to_list()
            
            return X
    
    def get_feature_names(self):
        # get back a feature column names
        return self.columns_list
    


class CustomizedStandardScaler(TransformerMixin):
    """ StandardScaler to return a pandas dataframe"""

    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled



class CategoricalAttributesTransformer(BaseEstimator, TransformerMixin):
    """
    Transform a categorical and oridinal attributes before training
    
    Parameters
    ----------
    categorial_attributes : list
        List of categorical attributes which need to transform
    oridinal_attribute: list or str
        Oridinal attributes to transform
    oridinal_value_transform : bool, default=True
        If True, apply transformation to oridinal attributes
    OH_encoder:  bool, default=True
        Apply One Hot Encoding to categorical attributes

    """
    
    def __init__(self, categorial_attributes, oridinal_attribute, oridinal_value_transform=True, OH_encoder=True ):
        self.categorial_attributes = categorial_attributes
        self.oridinal_attribute = oridinal_attribute
        self.OH_encoder = OH_encoder
        self.oridinal_value_transform = oridinal_value_transform
        
    
    def fit(self, X, y=None):
        return self # nothing else to do
    
    def transform(self, X, y=None):
        
        if self.oridinal_value_transform:
            # Divide age into sub categories
            bins = [10,20,30,40,50,60,70,80,90,100]
            labels = ['10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-100']
            age_group=pd.cut(X[self.oridinal_attribute], bins=bins, labels=labels)
            
            #inserting the age group after age and deleting it
            X.insert(1,self.oridinal_attribute + '_group', age_group)
            
            #dropping age column
            X.drop(self.oridinal_attribute, axis=1, inplace=True)
        
        
        if self.OH_encoder:
            
            self.categorial_attributes.append(self.oridinal_attribute + '_group')
            
            one_hot_encodded_X = pd.get_dummies(X[self.categorial_attributes], drop_first=True)
            encoded_X = pd.concat([X, one_hot_encodded_X], axis=1)
            encoded_X = encoded_X.drop(self.categorial_attributes, axis=1)
            
            # get the column names after transforming
            self.columns_list = encoded_X.columns.tolist()
            return encoded_X
        
    def get_feature_names(self):
        # get back the column names
        return self.columns_list