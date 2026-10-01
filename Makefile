.PHONY: install test run format format-check lint quality docker-build docker-run

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -v

run:
	python analysis.py

format:
	black analysis.py exploratory_data_analysis.py tests

format-check:
	black --check analysis.py exploratory_data_analysis.py tests

lint:
	flake8 analysis.py exploratory_data_analysis.py tests

quality: format-check lint test

docker-build:
	docker build -t dating-app-analysis .

docker-run:
	docker run --rm dating-app-analysis