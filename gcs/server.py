"""Ground-control-station panel for the GNSS-denied visual navigation runs (live or replay).

Serves the telemetry written by sim/visual_gps.py (outputs/sim_runs/<tag>/telemetry.jsonl + frames/) and a
satellite map of the operation area.

    python gcs/server.py            -> http://127.0.0.1:8050
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.geo import meters_per_degree  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402

RUNS = ROOT / "outputs" / "sim_runs"
WORLDS = ROOT / "sim" / "gz" / "worlds"
CACHE = ROOT / "outputs" / "gcs_cache"
app = FastAPI(title="GNSS-denied nav GCS")


def run_dir(tag: str) -> Path:
    d = (RUNS / tag).resolve()
    if RUNS.resolve() not in d.parents or not (d / "telemetry.jsonl").exists():
        raise HTTPException(404, "unknown run")
    return d


def run_world(tag: str) -> str:
    meta = run_dir(tag) / "run.json"
    if meta.exists():
        return json.loads(meta.read_text())["world"]
    return "visloc03_real_targets" if (WORLDS / "visloc03_real_targets.json").exists() else "visloc03_real"


@app.get("/")
def index():
    return FileResponse(ROOT / "gcs" / "index.html")


@app.get("/api/runs")
def runs():
    items = [d for d in RUNS.iterdir() if (d / "telemetry.jsonl").exists()]
    items.sort(key=lambda d: (d / "telemetry.jsonl").stat().st_mtime, reverse=True)
    return [dict(tag=d.name, frames=(d / "frames").exists(),
                 updated=(d / "telemetry.jsonl").stat().st_mtime) for d in items]


@app.get("/api/run/{tag}/info")
def info(tag: str):
    world = run_world(tag)
    w = json.loads((WORLDS / f"{world}.json").read_text())
    m_lat, m_lon = meters_per_degree(w["lat0"])
    half = w["extent_m"] / 2
    targets = [[w["lat0"] + t["north_m"] / m_lat, w["lon0"] + t["east_m"] / m_lon, t["cls"]] for t in w.get("targets", [])]
    return dict(world=world, lat0=w["lat0"], lon0=w["lon0"], camera=w["camera"], targets=targets,
                bounds=[[w["lat0"] - half / m_lat, w["lon0"] - half / m_lon], [w["lat0"] + half / m_lat, w["lon0"] + half / m_lon]],
                map_url=f"/api/map/{world}.jpg")


@app.get("/api/run/{tag}/telemetry")
def telemetry(tag: str, start: int = 0):
    lines = (run_dir(tag) / "telemetry.jsonl").read_text().splitlines()
    out = []
    for ln in lines[start:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:            # a line still being written (live run)
            break
    return JSONResponse(dict(start=start, next=start + len(out), rows=out))


@app.get("/api/run/{tag}/frame/{name}")
def frame(tag: str, name: str):
    f = (run_dir(tag) / "frames" / name).resolve()
    if f.parent != (run_dir(tag) / "frames").resolve() or not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/jpeg")


@app.get("/api/map/{world}.jpg")
def map_image(world: str):
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{world}.jpg"
    if not out.exists():
        w = json.loads((WORLDS / f"{world}.json").read_text())
        img = VisLocFlight(w["flight"]).sat.crop(w["lat0"], w["lon0"], w["extent_m"], w["extent_m"] / 2000)
        cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return FileResponse(out, media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="warning")
