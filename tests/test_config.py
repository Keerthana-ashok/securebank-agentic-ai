from securebank.config import Settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("SECUREBANK_MODEL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.model == "claude-opus-5-5"
    assert settings.effort == "medium"


def test_env_override(monkeypatch):
    monkeypatch.setenv("SECUREBANK_MODEL", "claude-sonnet-5-5")
    assert Settings(_env_file=None).model == "claude-sonnet-5-5"
