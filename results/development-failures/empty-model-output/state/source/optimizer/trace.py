"""Append structured events; callers supply only credential-free artifacts."""
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


class Trace:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()

    def artifact(self, relative):
        path = (self.output / relative).resolve()
        if not path.is_relative_to(self.output):
            raise ValueError("Artifact must stay inside output directory")
        return path.relative_to(self.output).as_posix()

    def record(self, event, *, artifacts=(), **fields):
        row = dict(fields)
        row.update(event=event, event_id=str(uuid.uuid4()),
                   timestamp=datetime.now(timezone.utc).isoformat(),
                   elapsed_ms=round((time.monotonic() - self.started) * 1000),
                   artifacts=[self.artifact(p) for p in artifacts])
        with (self.output / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
