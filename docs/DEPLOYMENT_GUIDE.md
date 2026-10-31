# Deployment guide

## 1. Local development (Windows PowerShell)

```powershell
cd "C:\college study\PR\backend"
python3.12 -m venv .venv                      # once
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item ..\.env.example .env                # once; then set GEMINI_API_KEY, APP_SECRET_KEY, ADMIN_API_TOKEN
python -m scripts.migrate                     # applies migrations (backs up data\app.db first)
python -m scripts.ingest_constitution         # parse + index + graph (also runs automatically at startup)
python -m scripts.check_gemini                # verifies key, models, streaming, embeddings
uvicorn app.main:app --port 8000
```

Open http://localhost:8000. On first start the app migrates the database, verifies the Constitution PDF edition and checksum, parses it into provisions and chunks, and builds the knowledge graph (about 10 s). Embeddings are generated in the background.

macOS/Linux: use `source .venv/bin/activate` and `cp ../.env.example .env`.

Optional:

```powershell
python -m scripts.embed_chunks                # (re)generate embeddings; resumes after quota limits
python -m scripts.fetch_statutes              # download official BNS/BNSS/Bonded Labour PDFs for verification
python -m scripts.seed_graph                  # rebuild graph and record statute verification
python -m scripts.inspect_graph               # integrity report (exit code 1 if unhealthy)
python -m pytest                              # 138 tests; live Gemini test: $env:RUN_LIVE_GEMINI=1
python -m scripts.run_eval                    # live evaluation -> docs\EVALUATION_REPORT.md
```

## 2. Docker Compose (backend + Neo4j)

```powershell
cd "C:\college study\PR"
Copy-Item .env.example backend\.env           # set GEMINI_API_KEY, APP_SECRET_KEY, ADMIN_API_TOKEN
$env:NEO4J_PASSWORD = "<choose a password>"   # used by both containers
docker compose up -d --build
docker compose ps                             # both services should be "healthy"
```

* `neo4j` (5.26 community) has a health check via `cypher-shell`, a persistent volume `neo4j-data`, and is bound to 127.0.0.1:7474/7687.
* `backend` starts after Neo4j is healthy. It uses `/data` (volume `app-data`) for SQLite, uploads and backups, and mounts `backend/data/statutes` read-only. It runs as a non-root user and has a `/health/ready` health check. On start it migrates, ingests, builds the graph, then syncs Neo4j and embeds in the background.
* Set `ENABLE_EMBEDDINGS=false` in the shell before `docker compose up` to skip embedding calls (for example on a free-tier key).
* Set `INSTALL_OCR=false` to build without Tesseract (image-upload OCR then returns 503; PDFs still work).
* Set `APP_HOST_PORT` to publish the backend on another host port if 8000 is taken.
* Status: the Neo4j service was verified; the backend image build could not complete in the verification environment because of network timeouts (TESTING_REPORT.md section 4).

Check:

```powershell
curl http://localhost:8000/health/ready
curl -H "X-Admin-Token: <ADMIN_API_TOKEN>" http://localhost:8000/api/health/llm
curl -H "X-Admin-Token: <ADMIN_API_TOKEN>" http://localhost:8000/api/admin/graph/inspect
```

Neo4j Browser: http://localhost:7474 (user `neo4j`).

## 3. Configuration and secrets

* All configuration is environment-driven (`.env.example` lists every variable). `backend/.env` is excluded from git and from the Docker build context.
* `GEMINI_API_KEY` is read only by the backend; it never appears in API responses, logs (the log formatter redacts `AIza…` keys and credential-like fields) or traces.
* Production: set `APP_ENV=production`, a long random `APP_SECRET_KEY`, and `ADMIN_API_TOKEN`; restrict `CORS_ORIGINS`; put the app behind HTTPS. `/health/ready` lists configuration problems.
* Session tokens are signed with `APP_SECRET_KEY`; rotating it signs everyone out (the UI obtains a new session automatically).

## 4. Database migrations, backup and rollback

* Migrations are versioned (`schema_migrations`), additive and non-destructive (`ALTER TABLE … ADD COLUMN`, new tables, indexes). No user data is deleted. Derived data (Constitution provisions, chunks, graph) is rebuilt by ingestion.
* **SQLite**: before any pending migration the database is copied with SQLite's online backup API to `data/backups/<db>-<UTC timestamp>-pre-v<N>.db`.

  ```powershell
  python -m scripts.migrate --status
  python -m scripts.rollback_db --list
  # stop the app first, then:
  python -m scripts.rollback_db --backup data\backups\app-20261030T120000Z-pre-v1.db
  ```

* **PostgreSQL** (supported through SQLAlchemy; install a driver such as `psycopg[binary]`): run `pg_dump -Fc` before upgrading; roll back with `pg_restore --clean`.
* **Neo4j** is a projection: it can always be rebuilt with `python -m scripts.sync_neo4j`. To reset it, `docker compose down` and remove the `neo4j-data` volume.

## 5. Updating the Constitution source

1. Download the latest English edition from the Legislative Department (https://legislative.gov.in/constitution-of-india/).
2. Replace `backend/data/constitution/constitution.pdf`.
3. Update `EDITION_MARKERS`, `EDITION_VERSION` and the publication date in `app/knowledge/ingestion.py` to the new edition's cover/preface wording. Until then the new file is ingested but marked `pending` and answers say the edition is unverified.
4. `python -m scripts.ingest_constitution --embed` and `python -m pytest tests/test_article_accuracy.py`.

## 6. Free-tier Gemini limits (observed 30 October 2026)

| Resource | Limit observed | Effect |
|---|---|---|
| `gemini-3.6-flash`, `gemini-3.5-flash` generation | 20 requests/day/model | After 20 requests the provider switches to `gemini-3.5-flash-lite` until the quota resets. |
| `gemini-embedding-001` | 1 000 requests/day, counted per text | A full index (1 688 chunks) needs two days on the free tier. |

Set `GEMINI_CLASSIFIER_MODEL=gemini-3.5-flash-lite` so classification does not use the primary model's quota. On a paid key these limits do not apply.
