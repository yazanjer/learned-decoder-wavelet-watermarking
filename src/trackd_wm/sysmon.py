"""GPU telemetry (utilisation and memory) sampled with nvidia-smi, plus per-method time windows."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List


class GPUSampler:
    def __init__(self, out_csv: Path, interval: float = 5.0):
        self.out_csv, self.interval = Path(out_csv), interval
        self._stop = threading.Event()
        self._t = None
        self.available = shutil.which("nvidia-smi") is not None

    def _run(self):
        new = not self.out_csv.exists()
        with open(self.out_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["t", "util_pct", "mem_used_mb"])
            while not self._stop.is_set():
                try:
                    r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                    u, m = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")]
                    w.writerow([f"{time.time():.1f}", u, m])
                    f.flush()
                except Exception:
                    pass
                self._stop.wait(self.interval)

    def start(self):
        if self.available:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=15)


class Windows:
    """Start/end time of each method, to attribute telemetry and cost."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.items: List[Dict] = json.loads(self.path.read_text()) if self.path.exists() else []

    def start(self, name: str, phase: str):
        self.items.append({"name": name, "phase": phase, "start": time.time(), "end": None})
        self._save()

    def end(self, status: str = "ok", **extra):
        self.items[-1].update(end=time.time(), status=status, **extra)
        self._save()

    def _save(self):
        self.path.write_text(json.dumps(self.items, indent=1))


def summarise(csv_path: Path, start: float, end: float) -> Dict[str, float]:
    rows = []
    if Path(csv_path).exists():
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                t = float(r["t"])
                if start <= t <= end:
                    rows.append((float(r["util_pct"]), float(r["mem_used_mb"])))
    if not rows:
        return {"samples": 0}
    u = [a for a, _ in rows]
    m = [b for _, b in rows]
    return {"samples": len(rows), "util_mean_pct": sum(u) / len(u), "util_median_pct": sorted(u)[len(u) // 2],
            "mem_peak_mb": max(m)}
