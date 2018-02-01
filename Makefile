# Run unit tests
test:
	python3 -m unittest discover -s tests

# Check test coverage using unittest module
coverage:
	coverage run --source=solver -m unittest discover -s tests; coverage report

# Check test coverage and show the results in browser
html:
	coverage run --source=solver -m unittest discover -s tests; coverage html; python -m webbrowser "./htmlcov/index.html" &

.PHONY: test