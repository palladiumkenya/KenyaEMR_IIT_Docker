install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint: 
	pylint --disable=R,C src/ tests/ pipelines/

test:
	PYTHONPATH=. pytest -vv tests/ --cov=src --cov-report=term-missing tests/

format:
	black src/ tests/