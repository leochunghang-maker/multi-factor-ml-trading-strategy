.PHONY: install install-loose lock test smoke signals paper paper-data paper-data-check paper-status paper-backup paper-daily paper-review dashboard report health significance data-check research doc-metrics

install:
	pip install --require-hashes -r requirements.lock

install-loose:
	pip install -r requirements.txt

lock:
	uv pip compile requirements.txt --python-version 3.12 --universal --generate-hashes -o requirements.lock

test:
	pytest -q

smoke:
	python scripts/run_smoke_tests.py

signals:
	python src/live/generate_live_signals.py

paper:
	python scripts/run_daily_paper_trading.py --skip-data-refresh

paper-data:
	python src/data_loader.py --paper

paper-data-check:
	python src/check_data.py --paper

paper-status:
	python scripts/paper_status.py

paper-backup:
	python scripts/paper_backup.py backup

paper-daily:
	python scripts/run_paper_daily.py

paper-review:
	python scripts/paper_weekly_review.py

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

research:
	python scripts/run_research_pipeline.py

doc-metrics:
	python src/check_doc_metrics.py
