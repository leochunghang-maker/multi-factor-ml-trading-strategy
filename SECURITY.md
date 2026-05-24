# Security Policy

This project is designed for local quantitative research and simulated paper trading only. It does
not enable live-money trading by default.

## Secrets Handling

- Never commit API keys, secret keys, passwords, access tokens, broker credentials, or private keys.
- Store local credentials in `.env`, which is ignored by Git.
- Keep `.env.example` limited to placeholders so reviewers can see required variable names without
  exposing real credentials.
- Do not paste credentials into notebooks, reports, logs, screenshots, commit messages, issues, or
  pull requests.

## Paper Trading Only

- The included workflow is intended for local research, validation, and simulated paper trading.
- Do not use this repository for live-money trading without adding separate production controls,
  broker-side safeguards, key management, monitoring, and compliance review.
- The paper executor should use paper endpoints and dry-run behavior unless deliberately reviewed and
  changed outside this research project.

## If a Key Is Exposed

1. Revoke or delete the exposed key immediately in the provider dashboard.
2. Create a new key only after the exposed key is disabled.
3. Remove the secret from local files, logs, screenshots, and any published artifacts.
4. If the secret was committed, rotate it even if the commit is later removed.
5. Review recent account activity for unexpected access or orders.
6. Update `.env` locally with the new key and keep `.env.example` as placeholders only.

## Reporting Security Issues

If you find a security issue in this project, document the affected file or workflow without
including secret values. Treat broker credentials and API keys as sensitive even for paper-trading
accounts.
