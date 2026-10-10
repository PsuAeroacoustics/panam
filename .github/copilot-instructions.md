# Copilot instructions for PANAM

Follow `AGENTS.md` at the repository root: setup, commands, the module map,
conventions, tests and the pull-request workflow. Human-oriented detail is in
`docs/DEVELOPING.md`. The rules that matter most, repeated from `AGENTS.md`:

## Never do

- Never commit data, results or machine paths: no `local_paths.toml`, no flight-test recordings, spheres or databases, no absolute paths from your machine, no credentials. Never `git add -f` an ignored file (`example_data/*.mat` and `mil_std_1474e_table_c1_full.csv` are tracked on purpose; add no others).
- Never commit flight-test data or results from the private validation repository, or cite its file paths. PANAM is public: restate the fact inline.
- Never prefix commands with `PYTHONPATH=.`; run from the repository root.
- Never change behavior or results without a test; never regenerate the golden results to make a refactor pass.
- Never push to `master`; use a branch and a pull request.
