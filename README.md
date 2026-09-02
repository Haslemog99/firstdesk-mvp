# FirstDesk

AI-driven workplace simulator. Practise a real first day in a job — Outlook, Slack, tools, deadlines, AI coworkers — and get scored on the dimensions that actually matter at work.

## Stack
- FastAPI + Pydantic v2
- LangChain + LangGraph multi-agent supervisor over Fireworks DeepSeek
- LangSmith tracing (optional — set `LANGSMITH_API_KEY`)
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
| `FIREWORKS_MODEL` | Optional. Default `deepseek-v4-pro-0813` |

Optional: `LANGSMITH_API_KEY` (traces LLM runs at https://smith.langchain.com), `HUME_API_KEY`, `ADMIN_TOKEN`, `STRICT_ENV=1` (fail fast).

## Password reset
Users reset their own password from the sign-in card via **Forgot password?**. The
server mails a single-use link that expires in 60 minutes; only a sha256 of the
token is stored, in the `password_resets` collection, swept by a TTL index.

| Var | Purpose |
|---|---|
| `RESEND_API_KEY` | Resend key. **Leave empty in dev** and the link is written to the log instead of sent |
| `RESEND_FROM` | Sender. Must be on a domain verified in Resend or sending returns 403 |
| `RESEND_REPLY_TO` | Optional monitored inbox for replies |
| `PUBLIC_BASE_URL` | Origin used to build the link. Falls back to `RAILWAY_PUBLIC_DOMAIN`, then the request host |

To unlock someone who cannot receive email, reset their password directly:
```bash
python3 scripts/reset_password.py --email someone@example.com --dry-run   # confirm the account
python3 scripts/reset_password.py --email someone@example.com             # prints a new password once
```
The script reads `MONGO_URI` from `.env`, looks the address up case-insensitively,
and kills any pending reset links for that account. Hand the password over on a
trusted channel and ask them to change it after signing in.

## LangChain / LangSmith CLI and MCP
- CLI: `langsmith` (Homebrew `langchain-ai/tap/langsmith-cli`) and `langchain` (`uv tool install langchain-cli`). Auth: `LANGSMITH_API_KEY` in `.env`, or `langsmith auth login`.
- Cursor MCP: project `.cursor/mcp.json` — **langsmith** (stdio via `scripts/langsmith-mcp.sh`, reads `.env`) and **langchain-docs** (`https://docs.langchain.com/mcp`). Reload MCP in Cursor after this change.
- FirstDesk workplace agents do **not** call MCP. Their tools stay the in-sim set (inbox, tasks, SIEM, …). MCP here is for Cursor / CLI only.

## Production checklist
- Set `STRICT_ENV=1`
- Set strong `ADMIN_TOKEN`
- Leave `DEMO_PASSWORD` empty (disables the demo login)
- Set `RESEND_API_KEY` and `PUBLIC_BASE_URL`, and add the public host to `ALLOWED_ORIGINS`
- Terminate TLS at a reverse proxy (Caddy / Nginx / Cloudflare) — HSTS is sent
- Restrict MongoDB Atlas IP allowlist to the deploy host
- Rotate any keys that ever sat in plaintext

## License
Proprietary — all rights reserved.
