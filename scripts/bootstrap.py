"""Incremental one-command setup: install, prepare, analyze/train only when stale."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / ".setup"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PIPELINE_VERSION = "3.0-incremental"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_parts(parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8")); h.update(b"\0")
    return h.hexdigest()


def run(args):
    print("\n$", " ".join(map(str, args)), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    SETUP.mkdir(exist_ok=True)
    requirements = ROOT / "requirements.txt"
    req_hash = sha256_file(requirements)
    deps_marker = SETUP / "requirements.json"
    deps_state = load_json(deps_marker)
    # A lightweight import check catches a damaged/incomplete virtual environment.
    import_check = "import numpy,pandas,scipy,sklearn,matplotlib,librosa,soundfile,joblib,streamlit,plotly"
    deps_ok = False
    if deps_state.get("requirements_sha256") == req_hash:
        check = subprocess.run([str(VENV_PY), "-c", import_check], cwd=ROOT)
        deps_ok = check.returncode == 0
    if not deps_ok:
        print("\n[SETUP] Dependencies changed or environment incomplete; installing once...", flush=True)
        run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(VENV_PY), "-m", "pip", "install", "-r", str(requirements)])
        run([str(VENV_PY), "-c", import_check])
        save_json(deps_marker, {"requirements_sha256": req_hash, "python": sys.version})
    else:
        print("[FAST] Dependencies unchanged; skipping pip.")

    raw = ROOT / "data" / "raw"
    ready_marker = SETUP / "mdvr_kcl_ready.json"
    ready = load_json(ready_marker)
    if not ready.get("ready"):
        print("\n[DATA] First-time dataset preparation (download may be ~606 MB).", flush=True)
        run([str(VENV_PY), "scripts/prepare_data.py"])
        wavs = list(raw.rglob("*.wav"))
        if not wavs:
            raise RuntimeError("Dataset preparation completed but no WAV files were found.")
        # This manifest is generated once and reused on subsequent launches.
        manifest = []
        for p in sorted(wavs):
            st = p.stat()
            manifest.append((p.relative_to(raw).as_posix(), st.st_size))
        manifest_hash = digest_parts([PIPELINE_VERSION, *[f"{n}:{s}" for n, s in manifest]])
        save_json(ready_marker, {"ready": True, "wav_count": len(wavs), "manifest_sha256": manifest_hash})
        print(f"[DATA] Cached dataset manifest for {len(wavs)} WAV files.")
    else:
        print(f"[FAST] Dataset ready ({ready.get('wav_count', 'cached')} WAV files); skipping download/extraction scan.")

    uci_csv = raw / "ReplicatedAcousticFeatures-ParkinsonDatabase.csv"
    uci_script = ROOT / "scripts" / "analyze_uci_features.py"
    uci_fingerprint = digest_parts([PIPELINE_VERSION, sha256_file(uci_csv), sha256_file(uci_script)])
    uci_marker = SETUP / "uci_analysis.json"
    if load_json(uci_marker).get("fingerprint") != uci_fingerprint:
        print("\n[ANALYSIS] UCI-489 analysis is missing or stale; regenerating graphs/tables.", flush=True)
        run([str(VENV_PY), "scripts/analyze_uci_features.py"])
        save_json(uci_marker, {"fingerprint": uci_fingerprint})
    else:
        print("[FAST] UCI-489 graphs/tables unchanged; skipping analysis.")

    train_script = ROOT / "scripts" / "train.py"
    audio_script = ROOT / "src" / "neuromargin" / "audio.py"
    train_fingerprint = digest_parts([
        PIPELINE_VERSION, req_hash, ready.get("manifest_sha256", ""),
        sha256_file(train_script), sha256_file(audio_script)
    ])
    train_marker = SETUP / "voice_model.json"
    model_file = ROOT / "models" / "voice_model.joblib"
    force = os.environ.pop("NEUROMARGIN_FORCE_RETRAIN", "") == "1"
    trained = load_json(train_marker)
    if force or not model_file.exists() or trained.get("fingerprint") != train_fingerprint:
        print("\n[TRAIN] Model missing or inputs changed; running participant-disjoint training.", flush=True)
        run([str(VENV_PY), "scripts/train.py"])
        if not model_file.exists():
            raise RuntimeError("Training finished without creating models/voice_model.joblib")
        save_json(train_marker, {"fingerprint": train_fingerprint})
    else:
        print("[FAST] Model and training inputs unchanged; skipping feature extraction and training.")

    print("\n[READY] Starting NeuroMargin dashboard. Close the browser tab/window to stop it.", flush=True)
    run([str(VENV_PY), "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: A setup step failed (exit code {e.returncode}).", file=sys.stderr)
        sys.exit(e.returncode or 1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
