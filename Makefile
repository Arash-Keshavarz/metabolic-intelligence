.PHONY: help venv install simulate preprocess train finetune api mlflow clean

PYTHON ?= python
VENV_DIR ?= .venv

help:
	@echo "Targets:"
	@echo "  venv        - create local virtualenv"
	@echo "  install     - install python dependencies"
	@echo "  simulate    - generate synthetic raw dataset"
	@echo "  preprocess  - build processed dataset artifacts"
	@echo "  train       - train global model (MLflow tracked)"
	@echo "  finetune    - per-user fine-tuning + comparison table"
	@echo "  mlflow      - run MLflow UI locally"
	@echo "  api         - run FastAPI service locally"
	@echo "  clean       - remove local artifacts"

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install:
	$(VENV_DIR)/bin/pip install -U pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

simulate:
	$(VENV_DIR)/bin/python -m src.data.simulate --config configs/simulate.yaml

preprocess:
	$(VENV_DIR)/bin/python -m src.data.preprocess --config configs/preprocess.yaml

train:
	$(VENV_DIR)/bin/python -m src.training.train --config configs/train.yaml

finetune:
	$(VENV_DIR)/bin/python -m src.training.finetune --config configs/finetune.yaml

mlflow:
	mkdir -p mlruns
	MLFLOW_TRACKING_URI=file:./mlruns $(VENV_DIR)/bin/mlflow ui --host 0.0.0.0 --port 5000

api:
	$(VENV_DIR)/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf data/processed artifacts mlruns
