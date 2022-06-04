from kaggle.api.kaggle_api_extended import KaggleApi
import os
import yaml

def get_data_from_kaggle(competition_name, file_name, path_to_save):
    api = KaggleApi()
    api.authenticate()
    api.competition_download_file(competition_name,file_name, path=path_to_save)

# Function to load yaml configuration file
def load_config(config_path, config_name):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config