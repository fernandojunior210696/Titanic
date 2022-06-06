## Create venv
create_venv:
	python3 -m venv venv

## Generate requirements
requirements:
	pip freeze > requirements.txt

## Download datasets
dataset:
	python3 src/data/make_dataset.py

## Train model
train:
	python3 src/models/train_model.py

## Submit results
submit:
	python3 src/models/predict_model.py

## Setup
setup:
	pip install -r requirements.txt