.PHONY: install test smoke signals paper dashboard report health significance data-check

install:
	pip install -r requirements.txt

test:
	pytest -q

smoke:
	python scripts/run_smoke_tests.py

signals:
	python src/live/generate_live_signals.py

paper:
	python scripts/run_daily_paper_trading.py --skip-data-refresh

dashboard:
	streamlit run dashboard/app.py

report:
	python src/generate_report.py

health:
	python src/operations/health.py

significance:
	python src/run_significance_report.py

data-check:
	python src/check_data.py --write-report
