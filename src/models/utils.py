import os
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd

# Function to load yaml configuration file
def load_config(config_path, config_name):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config

# Function to cast data as category
def column_to_category(pandas_series):
    cat_series = pandas_series.astype("category")
    return cat_series

# Function to cast data as category
def extract_titles_from_names(pandas_series):
    title_series = pandas_series.str.extract(' ([A-Za-z]+)\.', expand=False)
    title_series = title_series.str.replace('^(?!.*(Mr|Miss|Mrs)).*$', 'Master', regex=True)
    return title_series

# Class to do feature engineering
class PreProcessingData(BaseEstimator, TransformerMixin):
    """Apply some preprocessing in the dataframe"""
    
    def __init__(self, columnsNames):
        self.columnsNames = columnsNames

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X
        columnsNames = self.columnsNames
        
        df = pd.DataFrame(X, columns=columnsNames)
        
        # adjust features types
        df['Pclass'] = column_to_category(df['Pclass'])
        df['Sex'] = column_to_category(df['Sex'])
        df['Embarked'] = column_to_category(df['Embarked'])
        
        # Let's drop PassengerId, since it'll be not relevant for the modeling
        df.drop(['PassengerId'], inplace = True, axis=1)
        
        # Cap max Fare value to treat outliers
        df['Fare'] = df['Fare'].clip(upper=165)
        
        # Feature engineering - Fare
        #creating the intervals that we need to cut each range of ages
        interval = (0, 1, 13, 50, 100, 165) 

        #Seting the names that we want use to the categorys
        cats = ['Free', 'Cheap', 'Mid', 'Expensive', 'Very Expensive']

        # Applying the pd.cut and using the parameters that we created 
        df['FareClass'] = pd.cut(df.Fare, interval, labels=cats, include_lowest=True)
        
        # Let's drop Fare
        df.drop(['Fare'], inplace = True, axis=1)
        
        # Feature engineering - Family Size
        # Define size of the family
        df['FamilySize'] = df['SibSp'] + df['Parch']

        #creating the intervals that we need to cut each range of ages
        interval = (0, 1, 2, 3, 10) 

        #Seting the names that we want use to the categorys
        cats = ['Single', 'Small Family', 'Mid Family', 'Large Family']

        # Applying the pd.cut and using the parameters that we created 
        df['FamilyClass'] = pd.cut(df.FamilySize, interval, labels=cats, include_lowest=True)
        
        # Let's drop SibSp and Parch
        df.drop(['SibSp', 'Parch', 'FamilySize'], inplace = True, axis=1)
        
        # Feature Engineering - Age
        #creating the intervals that we need to cut each range of ages
        interval = (0, 5, 12, 18, 25, 35, 60, 120) 

        #Seting the names that we want use to the categorys
        cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']

        # Applying the pd.cut and using the parameters that we created 
        df["AgeBand"] = pd.cut(df.Age, interval, labels=cats)
        
        # Let's drop Age
        df.drop(['Age'], inplace = True, axis=1)
        
        # Feature Engineering - Name
        df['Title'] = extract_titles_from_names(df['Name'])
        df['Title'] = column_to_category(df['Title'])
        
        # Let's drop Name
        df.drop(['Name'], inplace = True, axis=1)
        
        # Let's drop useless features
        df.drop(['Cabin', 'Ticket'], inplace = True, axis=1)
        
        return df
  
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

def features_imputer(df, numerical_imputer_method, categorical_imputer_method):

    # Numeric features
    num_features = list(df.select_dtypes(include=['int64', 'float64']).columns)
    numerical_imputer = Pipeline(steps=[
        ('imputer', numerical_imputer_method)])

    # Categorical features
    cat_features = list(df.select_dtypes(include=['object', 'category']).columns)
    categorical_imputer = Pipeline(steps=[
        ('imputer', categorical_imputer_method)])

    features_names = num_features+cat_features

    # Features Pipeline
    transformer = ColumnTransformer(
        transformers=[
            ('num', numerical_imputer, num_features),
            ('cat', categorical_imputer, cat_features)])
    
    return transformer, features_names

def categorial_features_encoder(categorical_features, categorical_encoder_method):

    # New Categorical features
    categorical_transformer = Pipeline(steps=[categorical_encoder_method])

    # New Features Pipeline
    encoder = ColumnTransformer(
        transformers=[
            ('features', categorical_transformer, categorical_features)])
    
    return encoder
    