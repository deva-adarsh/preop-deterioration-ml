train:
	python -m src.run_train --data-dir data --model-dir models --alert-rate-cap 0.05

test:
	pytest -q

app:
	streamlit run app/streamlit_app.py
