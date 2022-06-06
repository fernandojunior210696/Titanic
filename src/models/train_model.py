import os
import pandas as pd
from utils import load_config, run_exps_random_search, run_hyperopt_experiments
from classifiers import experiments_arg
from sklearn.model_selection import train_test_split
from joblib import dump
import logging

logging.getLogger().setLevel(logging.INFO)

# Read configs from yaml config file
logging.info('***** Reading configurations *****')
config = load_config('src/config/', 'my_config.yaml')

# Load train and test data
logging.info('***** Loading training data *****')
df = pd.read_csv(os.path.join(config["data_directory"], config["train_data_name"]))

# Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis = 1), 
                                                    df.Survived, 
                                                    test_size=config["test_size"], 
                                                    random_state=config["seed"])

feature_engineering_step, space = experiments_arg(X_train)
# Run experiments
classifier = run_hyperopt_experiments(X_train, y_train, X_test, y_test, feature_engineering_step, space)

# # Models to experiment
# models = [('Logistic Regression', classifier, logreg_grid)]

# # Run experiments
# train_final_score = run_exps_random_search(models, X_train, y_train, config["n_jobs"])
# final_score = train_final_score.groupby(['model']).mean().round(3).reset_index()
# print(final_score)

# Save fitted model
logging.info('***** Dumping Trained Model *****')
model_path = os.path.join(config["artifact_path"], config["model_name"])
dump(classifier, model_path)