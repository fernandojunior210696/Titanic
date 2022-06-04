from utils import get_data_from_kaggle, load_config

# Read configs from yaml config file
config = load_config('src/config/', 'my_config.yaml')

# Download train data
get_data_from_kaggle(config["competition_name"], config["train_data_name"], config["path_to_save_data"])

# Download test data
get_data_from_kaggle(config["competition_name"], config["test_data_name"], config["path_to_save_data"])