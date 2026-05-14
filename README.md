# FirstDesk

AI-driven workplace simulator. Practise a real first day in a job — Outlook, Slack, tools, deadlines, AI coworkers — and get scored on the dimensions that actually matter at work.

## Stack
- FastAPI + Pydantic v2
- LangGraph multi-agent supervisor over Fireworks DeepSeek
- MongoDB Atlas
- Single-file frontend (vanilla JS)

## Local setup
```bash
cp .env.example .env          # then fill in secrets
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run.sh
```
Then open http://localhost:8000.

## Required env
| Var | Purpose |
|---|---|
| `AUTH_SECRET` | HMAC key for signed tokens. Generate: `python3 -c "import secrets; print(secrets.token_hex(32))"` |
| `MONGO_URI` | MongoDB Atlas connection string |
| `FIREWORKS_API_KEY` | LLM provider |

Optional: `LANGCHAIN_API_KEY`, `HUME_API_KEY`, `ADMIN_TOKEN`, `STRICT_ENV=1` (fail fast).

## Production checklist
- Set `STRICT_ENV=1`
- Set strong `ADMIN_TOKEN`
- Leave `DEMO_PASSWORD` empty (disables the demo login)
- Terminate TLS at a reverse proxy (Caddy / Nginx / Cloudflare) — HSTS is sent
- Restrict MongoDB Atlas IP allowlist to the deploy host
- Rotate any keys that ever sat in plaintext

## License
Proprietary — all rights reserved.
