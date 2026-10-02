# GNSS-Denied Visual Navigation for Fixed-Wing UAVs

Localize a UAV **without GPS** by matching its downward camera against a satellite map, then feed the
position to the autopilot (ArduPilot) so the mission continues when GNSS is jammed.

> Status: **Sprint 1 done** (visual localization on real UAV imagery). Sprint 2 (closed-loop ArduPilot SITL +
> Gazebo with GPS jamming) in progress.

![match example](docs/match_example.jpg)
*Unseen test flight 11. Top: 291 consistent matches → confident fix. Bottom: season change between photo and
map (dry vs. ploughed fields), only 6 matches → the system reports "not confident" instead of a wrong position.*

## Pipeline

```
drone photo ──► north-up + metric scale ──► DINOv2 (fine-tuned) ──► top-10 map tiles ──► SuperPoint+LightGlue
              (compass heading, baro height)    global descriptor        FAISS search        + RANSAC similarity
                                                                                                    │
                                              lat/lon + confidence (#inliers) ◄─────────────────────┘
```

Only GNSS-free inputs are used at test time: the image, heading (compass/IMU) and height (barometer).
The whole satellite map of the flight area is searched (no position prior).

## Results — Sprint 1

Dataset: [UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) (real fixed-wing/multirotor flights over 11
Chinese regions, 0.3 m satellite maps). **Split by region** so test areas are never seen in training:

| split | flights | images |
|---|---|---|
| train | 01, 02 (Changjiang) · 05 (Yunnan) · 08, 09 (Huzhou) · 10 (Huailai) | 4 304 |
| val (model selection) | 06 (Zhuxi, mountains) | 344 |
| **test** | 03, 04 (Taizhou) · 11 (Shandan, arid plateau) | **2 096** |

Flight 07 is excluded (30 images, degenerate map bounds).

Test results (2 096 queries, global search over each flight's map, 11k–24k tiles per map):

| method | median error | < 50 m | < 100 m | confident | confident & > 100 m off |
|---|---|---|---|---|---|
| DINOv2-B pretrained (top-1 tile) | 199 m | 18.2 % | 36.9 % | – | – |
| DINOv2-B fine-tuned (top-1 tile) | 39 m | 63.1 % | 87.4 % | – | – |
| pretrained + LightGlue refine | 30 m | 69.1 % | 75.3 % | 73.6 % | 2 |
| **fine-tuned + LightGlue refine** | **24 m** | **88.7 %** | **97.3 %** | **94.8 %** | **2 / 1 987** |

Per test flight (fine-tuned + LightGlue): 03 → 18.5 m median, 04 → 32.1 m, 11 → 27.2 m.
"Confident" = at least 25 RANSAC inliers. Latency on RTX 5090: retrieval < 1 ms/query, LightGlue refine ≈ 0.55 s/query (10 candidates, unoptimized).

Fine-tuning: last 4 DINOv2 blocks, symmetric InfoNCE, in-batch negatives from the same city with GT positions
≥ 150 m apart (consecutive frames overlap and would otherwise be false negatives), haze/blur/colour and
±5° heading-noise augmentation. 15 epochs ≈ 3 minutes on one GPU; best epoch chosen on the validation flight.

### Findings worth knowing
- **Label time lag.** Refined errors are biased ~15 m *along the flight direction* with ~0 cross-track bias:
  the GT GPS stamp lags the photo by ~1 s. A large part of the remaining error is label noise, not the model.
  Any correction will be fit on training flights only.
- **`height` is above sea level**, not above ground. We calibrate an *effective* focal length per flight
  (`scripts/calibrate_camera.py`), valid while terrain height is roughly constant along the flight.
- **Heading convention.** Rotating the photo by `-Phi1` makes it north-up on every flight (residual 1–8°),
  recovered from data rather than assumed.

## Reproduce

```bash
pip install -r requirements.txt
# data: UAV-VisLoc into ../data/UAV-VisLoc (or set VISLOC_ROOT); flight 09's four map tiles are merged into satellite09.tif
python scripts/calibrate_camera.py --flight 03          # per flight -> configs/camera_visloc03.json
python scripts/prepare.py 01,02,03,04,05,06,08,09,10,11 # north-up queries + metric map cache
bash   scripts/run_sprint1.sh                           # baseline, fine-tune, evaluation (+ LightGlue)
python scripts/visualize_match.py --flight 11 --idx 150 400
```

Simulator setup (WSL2 / Ubuntu 24.04): `sim/setup_root.sh` (Gazebo Harmonic + system packages, as root) then
`sim/setup_user.sh` (ArduPlane SITL + ardupilot_gazebo). `sim/smoke_sitl.sh` checks the MAVLink link.

## Roadmap
- [x] Sprint 1 — visual localization on real imagery (retrieval + fine-tuning + LightGlue, confidence)
- [ ] Sprint 2 — Gazebo world textured with the satellite map, downward camera, `GPS_INPUT` to ArduPlane, GPS-jamming scenario
- [ ] Sprint 3 — learned fusion with IMU (GRU vs. Kalman baseline), visual attitude estimation, runway detection for landing
- [ ] Sprint 4 — ONNX/TensorRT latency, demo video, model release
