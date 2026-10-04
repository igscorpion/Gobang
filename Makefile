PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

.PHONY: install backend frontend dev env train eval

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r web/backend/requirements.txt
	cd web/frontend && npm install

env:
	$(PY) -m rl.environment.gomoku

train:
	$(PY) -m rl.training.train

eval:
	$(PY) -m rl.evaluation.evaluate

backend:
	$(UVICORN) web.backend.main:app --reload --host 127.0.0.1 --port 8000

frontend:
	cd web/frontend && npm run dev

dev:
	@echo "Backend: http://127.0.0.1:8000"
	@echo "Frontend: http://127.0.0.1:5173"
	$(UVICORN) web.backend.main:app --reload --host 127.0.0.1 --port 8000 & \
	cd web/frontend && npm run dev
