"""Verify distributed evidence bytes offline; no model loading or network access."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "PROVENANCE.json").read_text(encoding="utf-8"))
seen = set()
for record in manifest["records"]:
    relative = record["public_path"]
    target = (root / relative).resolve()
    if root not in target.parents or relative in seen:
        raise ValueError("Invalid or duplicate manifest path")
    seen.add(relative)
    data = target.read_bytes()
    assert len(data) == record["public_bytes"], relative
    assert hashlib.sha256(data).hexdigest() == record["public_sha256"], relative
    if record["byte_preservation_required"]:
        assert record["source_sha256"] == record["public_sha256"], relative
print(f"PASS: {len(seen)} distributed source files; mandatory byte-preservation hashes agree")
