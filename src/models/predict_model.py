import pandas as pd
from joblib import load
from utils import submite_to_competition, load_config
import os

# Read configs from yaml config file
config = load_config('src/config/', 'my_config.yaml')

# Load model
model_path = os.path.join(config["artifact_path"], config["model_name"])
model = load(model_path)

# Read submission dataset
submission_df = pd.read_csv(os.path.join(config["data_directory"], config["test_data_name"]))

# Get predictions
results = pd.DataFrame()
results['PassengerId'] = submission_df.PassengerId
results['Survived'] = model.predict(submission_df)

# Save predictions
results.to_csv(config["submition_path"], index=False)

# Submit to competition
submite_to_competition(config["submition_path"], config["message_on_submition"], config["competition_name"])