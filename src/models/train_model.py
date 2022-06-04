import os
import pandas as pd
from utils import load_config, PreProcessingData, features_imputer, categorial_features_encoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # handle missing values (naive approach)
from sklearn.impute import KNNImputer # handle missing values (better approach)
from sklearn.preprocessing import OneHotEncoder # enconding cat variables
from sklearn.decomposition import PCA, TruncatedSVD # Principal Components Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA # Linear Discriminant Analysis
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

# Read configs from yaml config file
config = load_config('src/config/', 'my_config.yaml')

# Load train and test data
df = pd.read_csv(os.path.join(config["data_directory"], config["train_data_name"]))
submission_df = pd.read_csv(os.path.join(config["data_directory"], config["test_data_name"]))

# Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis = 1), 
                                                    df.Survived, 
                                                    test_size=config["test_size"], 
                                                    random_state=config["seed"])

# Imputation step
numerical_imputer_method = KNNImputer(n_neighbors=4, weights="uniform")
categorical_imputer_method = SimpleImputer(strategy='most_frequent')
features_imputer_step, col_features = features_imputer(X_train, numerical_imputer_method, categorical_imputer_method)

# Categorical encoder step
categorical_features = ['Pclass', 'Sex', 'Embarked', 'FareClass', 'FamilyClass', 'AgeBand', 'Title']
categorical_encoder_method = ('onehot', OneHotEncoder(handle_unknown='ignore'))
features_encoder_step = categorial_features_encoder(categorical_features, categorical_encoder_method)

# Feature Engineering Pipeline
feature_engineering_pipeline = Pipeline(steps=[
    ('features_preprocessor', features_imputer_step),
    ('preprocessing', PreProcessingData(columnsNames = col_features)),
    ('hot_encoder', features_encoder_step),
    ('dimensionality_reduction', TruncatedSVD())])

# Final Classifier Pipeline
feature_engineering_step = ('feature_engineering', feature_engineering_pipeline)
ml_classifier = ('logreg', LogisticRegression(n_jobs=config["n_jobs"], random_state = config["seed"]))
classifier = Pipeline(steps=[feature_engineering_step,ml_classifier])

# Train classifier
classifier.fit(X_train, y_train)

# Save fitted model
model_path = os.path.join(config["artifact_path"], config["model_name"])
dump(classifier, model_path) 