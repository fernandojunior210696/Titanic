from utils import load_config, PreProcessingData, features_imputer, categorial_features_encoder
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # handle missing values (naive approach)
from sklearn.impute import KNNImputer # handle missing values (better approach)
from sklearn.preprocessing import OneHotEncoder # enconding cat variables
from sklearn.decomposition import PCA, TruncatedSVD # Principal Components Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA # Linear Discriminant Analysis
from sklearn.linear_model import LogisticRegression
import logging

# Classification Models
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.svm import SVC # SVM
from sklearn.ensemble import RandomForestClassifier # RandomForest
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from xgboost import XGBClassifier # Xgboost

# Tunning bayesian optimization
from hyperopt import hp
from hyperopt.pyll.base import scope


logging.getLogger().setLevel(logging.INFO)

# Read configs from yaml config file
logging.info('***** Reading configurations *****')
config = load_config('src/config/', 'my_config.yaml')

def experiments_arg(X_train):
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
        ('dimensionality_reduction', TruncatedSVD())
        ])

    feature_engineering_step = ('feature_engineering', feature_engineering_pipeline)

    space = hp.choice('classifiers', [
    {
        'model':LogisticRegression(n_jobs=config["n_jobs"], random_state = config["seed"]),
        'params':{
            'hyper_param_groups' :hp.choice('hyper_param_groups_1', 
            [
                {
                'model__solver': hp.choice('solver_block1', ['newton-cg', 'sag', 'saga', 'lbfgs']),
                'model__penalty':hp.choice('penalty_block1', ['l2']),
                'model__fit_intercept': hp.choice('fit_intercept_block1', [True, False]),
                'model__C': hp.lognormal('C_block1', 0, 1.0),
                'feature_engineering__dimensionality_reduction__n_components': scope.int(hp.quniform('n_components_block1', 1, 28, 1))
                },
                {
                'model__solver': hp.choice('solver_block2', ['liblinear']),
                'model__penalty':hp.choice('penalty_block2', ['l2']),
                'model__fit_intercept': hp.choice('fit_intercept_block2', [True, False]),
                'model__C': hp.lognormal('C_block2', 0, 1.0),
                'feature_engineering__dimensionality_reduction__n_components': scope.int(hp.quniform('n_components_block2', 1, 28, 1))
                },
                {
                'model__solver': hp.choice('solver_block3', ['saga']),
                'model__penalty':hp.choice('penalty_block3', ['l1']),
                'model__fit_intercept': hp.choice('fit_intercept_block3', [True, False]),
                'model__C': hp.lognormal('C_block3', 0, 1.0),
                'feature_engineering__dimensionality_reduction__n_components': scope.int(hp.quniform('n_components_block3', 1, 28, 1))
                },
            ])
        }
    },

    {
        'model':XGBClassifier(n_jobs=config["n_jobs"], random_state = config["seed"]),
        'params':{
            'hyper_param_groups' :hp.choice('hyper_param_groups_2', 
            [
                {
                "model__objective": hp.choice("model__objective", ['binary:logistic', 'binary:logitraw', 'binary:hinge']),
                "model__learning_rate": hp.loguniform('model__learning_rate', np.log(0.001), np.log(0.3)),
                "model__colsample_bytree": hp.uniform('model__colsample_bytree', 0, 1),
                "model__colsample_bylevel": hp.uniform('model__colsample_bylevel', 0, 1),
                "model__colsample_bynode": hp.uniform('model__colsample_bynode', 0, 1),
                "model__subsample": hp.uniform('model__subsample', 0, 1),
                "model__max_depth": scope.int(hp.quniform('model__max_depth', 1, 25, 1)),
                "model__n_estimators": scope.int(hp.quniform('model__n_estimators', 1, 3000, 10)),
                "model__reg_alpha": hp.uniform('model__reg_alpha', 0, 1),
                "model__reg_lambda": hp.uniform('model__reg_lambda', 0, 1),
                "model__gamma": hp.uniform('model__gamma', 0, 30),
                'feature_engineering__dimensionality_reduction__n_components': scope.int(hp.quniform('n_components_block4', 1, 28, 1))
                }
            ])
        }
    },

    ])

    return feature_engineering_step, space