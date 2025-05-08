install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint: 
	pylint --disable=R,C src/

test:
	PYTHONPATH=. pytest -vv tests/