# Security Policy

This project is designed for local quantitative research and simulated paper trading only. It does
not enable live-money trading by default.

## Secrets Handling

- Never commit API keys, secret keys, passwords, access tokens, broker credentials, or private keys.
- The project needs no credentials: prices come from public Yahoo data and paper trading is a local
  simulated broker. If you keep any local secrets, put them in `.env`, which is ignored by Git.
- Do not paste credentials into notebooks, reports, logs, screenshots, commit messages, issues, or
  pull requests.

## Paper Trading Only

- The included workflow is intended for local research, validation, and simulated paper trading.
- Do not use this repository for live-money trading without adding separate production controls,
  broker-side safeguards, key management, monitoring, and compliance review.
- There is no broker connection or broker API client. Orders are filled by the local simulated broker
  (`src/execution/simulated_broker.py`) and never leave the machine. `tests/test_no_broker_connection.py`
  fails if a broker SDK, HTTP client or broker endpoint is added to `src/`, `scripts/` or `dashboard/`.

## If a Key Is Exposed

1. Revoke or delete the exposed key immediately in the provider dashboard.
2. Create a new key only after the exposed key is disabled.
3. Remove the secret from local files, logs, screenshots, and any published artifacts.
4. If the secret was committed, rotate it even if the commit is later removed.
5. Review recent account activity for unexpected access or orders.
6. Update `.env` locally with the new key.

## Reporting Security Issues

If you find a security issue in this project, document the affected file or workflow without
including secret values. Treat broker credentials and API keys as sensitive even for paper-trading
accounts.
