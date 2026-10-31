"""Model config: live default, automatic fallback, boot probe surfaced in /api/health."""


def test_default_model_is_not_a_retired_one(app_mod):
    retired = {"accounts/fireworks/models/deepseek-v4-pro",
               "accounts/fireworks/models/deepseek-v4-pro-0813"}
    assert app_mod.FIREWORKS_MODEL not in retired
    assert app_mod.FIREWORKS_FALLBACK_MODEL not in retired


def test_main_llm_has_a_fallback_on_a_different_model(app_mod):
    from langchain_core.runnables import RunnableWithFallbacks
    assert isinstance(app_mod.llm, RunnableWithFallbacks)
    models = [app_mod.llm.runnable.model_name] + [f.model_name for f in app_mod.llm.fallbacks]
    assert len(set(models)) == 2


def test_fallback_answers_when_primary_404s(app_mod, monkeypatch):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    primary, fallback = app_mod.llm.runnable, app_mod.llm.fallbacks[0]

    def dead(*_a, **_k):
        raise RuntimeError("404 model not found")

    def live(*_a, **_k):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="fallback ok"))])

    monkeypatch.setattr(type(primary), "_generate", lambda self, *a, **k: dead() if self is primary else live())
    assert app_mod.llm.invoke("hi").content == "fallback ok"
    assert fallback is not primary


def test_probe_reports_failure_and_health_degrades(app_mod, client, monkeypatch):
    def fake_probe(model_id):
        return "failed: NotFoundError: 404" if model_id == app_mod.FIREWORKS_MODEL else "ok"
    monkeypatch.setattr(app_mod, "_probe_one", fake_probe)
    status = app_mod.probe_llm_models()
    assert status["primary"].startswith("failed") and status["fallback"] == "ok"
    h = client.get("/api/health").json()
    assert h["status"] == "degraded"
    assert h["ai"] == {"primary": "failed", "fallback": "ok"}
    assert "404" not in str(h)  # provider error text stays out of a public endpoint


def test_health_ok_when_models_live(app_mod, client, monkeypatch):
    monkeypatch.setattr(app_mod, "_probe_one", lambda m: "ok")
    app_mod.probe_llm_models()
    assert client.get("/api/health").json()["status"] == "ok"
