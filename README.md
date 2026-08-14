# sse-db
Auto daily data of equities trading in the Shanghai Stock Exchange (SSE) into Mongodb Atlas

Data-only GitHub repo for the nightly updater

Traced by walking the actual import graph of scripts/update_daily.py, then built and tested: uv sync succeeds, all six retained scripts run, and the environment excludes every research dependency.

27 files, 460 KB. Everything else stays off GitHub.

1. What to upload
pyproject.toml              # keep BYTE-IDENTICAL to the research repo
uv.lock                     #   (see §3 — do not trim these)
.python-version
.gitignore
.env.example                # placeholders only; verified no real credentials

.github/workflows/
    update-daily.yml

src/sse/
    __init__.py             # no eager imports, so the closure stays small
    config.py
    logging_setup.py
    codec.py                # int32 x 10,000 + sha256 — every chunk goes through this
    calendar_utils.py       # needed by verify_integrity / backfill
    db/
        __init__.py
        client.py
        repository.py
        schema.py
    ingest/
        __init__.py
        provider.py
        quality.py
        universe.py

scripts/
    _path.py                # imported as bare `import _path` — easy to miss
    update_daily.py         # the scheduled job
    verify_integrity.py     # runs after each full refresh
    backfill.py             # for re-seeding or --missing-only
    discover_universe.py    # manual universe re-runs
    init_db.py              # collections + indexes
    ping_atlas.py           # connectivity check

reports/.gitkeep            # the run report is written here
Strictly required vs. merely useful

update_daily.py alone needs only these 13 files:

scripts/_path.py            src/sse/db/client.py
scripts/update_daily.py     src/sse/db/repository.py
src/sse/__init__.py         src/sse/db/schema.py
src/sse/config.py           src/sse/ingest/__init__.py
src/sse/logging_setup.py    src/sse/ingest/provider.py
src/sse/codec.py            src/sse/ingest/quality.py
src/sse/db/__init__.py      src/sse/ingest/universe.py

calendar_utils.py and the other five scripts are pulled in only by verify_integrity.py / backfill.py. Include them — being able to re-seed or re-verify from the repo without a local checkout is worth five small files.

No logs/.gitkeep: logging_setup creates the directory itself (verified), and .gitignore excludes logs/ anyway.

2. What to leave out
Not uploaded	Why
src/sse/research/	Nothing in the update path imports it
src/sse/api/	Ditto — the REST layer is not part of the scheduled job
scripts/run_backtest.py, analyse_correlation.py, measure_storage.py, purge_non_equity.py, doctor.py	Research / one-off maintenance
tests/, notebooks/, docs/, README.md, Makefile	Not executed by the workflow
reports/*.json	Already gitignored; the workflow uploads them as artifacts instead
3. Do not trim pyproject.toml or uv.lock

Tempting, but wrong. uv.lock is resolved across all dependency groups. Removing the research / tests / notebook groups from pyproject.toml forces uv to re-lock, producing a different uv.lock in each repo — and then "recreate the environment from uv.lock" (your own acceptance check) means two different things depending on which repo you're standing in.

Keep both files identical and trim at invocation time instead. pyproject.toml sets [tool.uv] default-groups = "all", so the workflow passes --no-default-groups:

yaml
run: uv sync --no-default-groups
run: uv run --no-default-groups python scripts/update_daily.py

--no-default-groups is needed on uv run too, not just uv sync — uv run re-syncs using the project's default groups and will silently reinstall everything otherwise. Verified: with the flag, yfinance/pymongo/numpy/pandas are present and scipy/sklearn/statsmodels/jupyterlab are absent.

4. Repo settings
Setting	Value
Secret SSE_MONGODB_URI	the mongodb+srv:// string, password percent-encoded
Variable SSE_DB_NAME	sse_market (optional; the workflow defaults to it)

Config is read from the environment via the SSE_ prefix, so no .env is needed, and SecretStr keeps the URI out of tracebacks and CI logs (verified).

Make the repo private. Not for the code — for the shape of the Atlas access list you are about to open.

5. The Atlas access-list problem

GitHub runners have rotating egress IPs. M0 supports neither VPC peering nor private endpoints, so this workflow requires opening the Atlas access list to 0.0.0.0/0, behind a database user scoped to read/write on sse_market only.

schedule_windows.ps1 avoids this entirely — one machine, one address, allow-listed precisely. On M0 with a fixed workstation that is the better choice, and the GitHub repo then becomes a backup/portability copy rather than the thing that runs the job.
