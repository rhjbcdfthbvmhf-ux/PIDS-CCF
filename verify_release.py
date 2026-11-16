"""Verify release integrity and regenerate recorded Tables 10 and 11 on CPU.

Only the Python standard library is required. No training or downloads occur.
"""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import shutil

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit("Release file is missing or changed: " + name)
    sources = sorted(ROOT / name for name in manifest["files"] if name.endswith(".py"))
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    with tempfile.TemporaryDirectory(prefix="pids_ccf_verify_") as directory:
        work = Path(directory)
        shutil.copytree(ROOT / "revision", work / "revision")
        subprocess.run([sys.executable, str(work / "revision/make_native_result_objects.py")], check=True)
        subprocess.run([sys.executable, str(work / "revision/make_published_result_objects.py"),
                        str(work / "revision/PUBLISHED_AUDIT_AND_RESULTS.json"),
                        str(work / "revision/result_objects")], check=True)
        for name, digest in manifest["expected_outputs"].items():
            actual = hashlib.sha256((work / "revision/result_objects" / name).read_bytes()).hexdigest()
            if actual != digest:
                raise SystemExit("Regenerated result differs: " + name)
    print(json.dumps({"status": "PASS", "verified_files": len(manifest["files"]),
                      "parsed_python_sources": len(sources),
                      "recomputed_outputs": list(manifest["expected_outputs"]),
                      "training_executed": False}, indent=2))


if __name__ == "__main__":
    main()
