PYTHON ?= python3
UV ?= uv

.PHONY: install backend frontend dev env train eval

install:
	$(UV) sync --python 3.11
	cd web/frontend && npm install

env:
	$(UV) run python -m rl.environment.gomoku

train:
	$(UV) run python -m rl.training.train

eval:
	$(UV) run python -m rl.evaluation.evaluate

backend:
	$(UV) run uvicorn web.backend.main:app --reload --host 127.0.0.1 --port 8000

frontend:
	cd web/frontend && npm run dev

dev:
	@echo "Backend: http://127.0.0.1:8000"
	@echo "Frontend: http://127.0.0.1:5173"
	$(UV) run uvicorn web.backend.main:app --reload --host 127.0.0.1 --port 8000 & \
	cd web/frontend && npm run dev
