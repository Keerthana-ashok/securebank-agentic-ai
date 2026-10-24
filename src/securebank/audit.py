"""Append-only JSON Lines audit log for actions agents take on a customer's behalf."""

import json
from datetime import UTC, datetime
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(self, **event) -> None:
        entry = {"ts": datetime.now(UTC).isoformat(), **event}
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
