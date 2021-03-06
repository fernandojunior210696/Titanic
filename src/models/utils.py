import os
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.model_selection import cross_validate
import logging
from sklearn.model_selection import RandomizedSearchCV
import mlflow

# Tunning bayesian optimization
from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll.base import scope
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, make_scorer

from kaggle.api.kaggle_api_extended import KaggleApi
def submite_to_competition(submition_path, message, competition_name):
    api = KaggleApi()
    api.authenticate()
    api.competition_submit(submition_path, message=message, competition=competition_name)


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
        ('imputer', numerical_imputer_method)
        ])

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

# compare various metrics of model
def get_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2)}

def create_mlflow_experiment(name):
    try:
         # Creates a new experiment
         experiment_id = mlflow.create_experiment(name)
    except:
        # Retrieves the experiment id from the already created project
        experiment_id = mlflow.get_experiment_by_name(name).experiment_id

    return experiment_id

# Hyperopt

def run_hyperopt_experiments(X_train, y_train, X_test, y_test, feature_engineering_step, space):
        
    # config logs
    logging.getLogger().setLevel(logging.INFO)
    logging.info('***** Step 1: Run Experiments Started *****')

    # silent process
    import warnings
    warnings.filterwarnings('ignore') 
    
    # objective funtion 
    def objective(args):

        # Initialize model pipeline
        model = Pipeline(steps=[
            feature_engineering_step,
            ('model', args['model']) # args[model] will be sent by fmin from search space
        ])

        # parameters from pipeline
        print(args['params'])
        model.set_params(**args['params']['hyper_param_groups'])

        score = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1, error_score='raise')
        print(f'Model Name: {args["model"]}: ', score)

        # return the negative mean of the eval metric
        return {
        'loss': -score.mean(),  
        'status': STATUS_OK
    }

    # The Trials object will store details of each iteration
    trials = Trials()

    # Run the hyperparameter search using the tpe algorithm
    best = fmin(objective,
                space,
                algo=tpe.suggest,
                max_evals=300,
                trials=trials)
        
    # Get the values of the optimal parameters
    best_params = space_eval(space, best)
    
    print('**** The best parameters are: {} ****'.format(best_params))

    # Initialize best model pipeline
    model = Pipeline(steps=[
        feature_engineering_step,
        ('model', best_params['model']) # args[model] will be sent by fmin from search space
    ])

    model.set_params(**best_params['params']['hyper_param_groups'])

    # Create mlflow experiment with model name
    name = "Hyperopt"
    experiment_id = create_mlflow_experiment(name)

    with mlflow.start_run(run_name = name, experiment_id=experiment_id):
        logging.info('Fitting best model from Hyperopt')

        # Fit the model with the optimal hyperparamters
        model.fit(X_train, y_train)

        # Predicting with the best model
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Get metrics
        y_true = y_train
        y_pred = y_pred_train
        run_metrics = get_metrics(y_true, y_pred)

        # Log metrics and parameters of experiment
        mlflow.log_metrics(run_metrics)
        mlflow.log_params(best_params['params']['hyper_param_groups'])

        # Classification Report 
        print('Training Classification Report for estimator: ',
            str(model).split('(')[0])
        print('\n', classification_report(y_train, y_pred_train))
        print('\n', classification_report(y_test, y_pred_test))

        mlflow.end_run()

    return model


def run_exps_random_search(models, X_train, y_train, pararell_threads):
        
    # config logs
    logging.getLogger().setLevel(logging.INFO)
    logging.info('***** Step 1: Run Experiments Started *****')

    logging.info("Starting experiment: {}".format(experiment_id))
    
    scoring_df = []
    final_scoring = {}

    # Loop over models configurations
    for name, model, model_grid in models:

        # Create mlflow experiment with model name
        experiment_id = create_mlflow_experiment(name)

        with mlflow.start_run(run_name = name, experiment_id=experiment_id):
            logging.info('Starting Random Search From: %s Model', name)
            
            # Initialize random search pipeline
            clf = RandomizedSearchCV(model, 
                                    model_grid, 
                                    random_state=42,cv=5, 
                                    scoring='accuracy',
                                    n_jobs=pararell_threads,
                                    return_train_score=True)

            # Fit random search pipeline
            search = clf.fit(X_train, y_train)

            # Get best parameters and refit
            best_params = search.best_params_
            model.set_params(**best_params)
            model.fit(X_train, y_train)

            # Get metrics
            y_true = y_train
            y_pred = model.predict(X_train)
            run_metrics = get_metrics(y_true, y_pred)

            # Log metrics and parameters of experiment
            mlflow.log_metrics(run_metrics)
            mlflow.log_params(best_params)

            logging.info('Cross Validating: %s Model', name)

            # Train scores
            final_scoring= cross_validate(model, 
                X_train, 
                y_train, 
                cv=5, 
                scoring={'accuracy': 'accuracy'},
                return_train_score=True,
                verbose = 30,
                n_jobs= pararell_threads)
            
            final_scoring = final_scoring
            
            # Get all models train scores
            this_df = pd.DataFrame(final_scoring).round(4)
            this_df['model'] = name
            scoring_df.append(this_df)
            train_final_score = pd.concat(scoring_df, ignore_index=True)
            mlflow.end_run()

        logging.info('***** Step 1: Run Experiments Finished *****')
            
        return train_final_score
