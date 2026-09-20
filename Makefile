install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -v

run:
	python analysis.py
