"""Generate a Gazebo Harmonic world whose ground is a georeferenced satellite map, plus a Zephyr with a nadir camera.

Ground: N x N square tiles (tile_m each) cut north-up from the flight's GeoTIFF, centered on (lat0, lon0).
Gazebo world frame is ENU (x = East, y = North), origin = (lat0, lon0) = ArduPilot home.

Output (sim/gz/):
  worlds/<name>.sdf
  models/satmap_<name>/{model.config, model.sdf, textures/*.png}
  models/zephyr_cam/{model.config, model.sdf}
  worlds/<name>.json   (origin + camera intrinsics, read by the bridge/localizer)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gdnav.geo import meters_per_degree  # noqa: E402
from gdnav.visloc import VisLocFlight  # noqa: E402

OUT = ROOT / "sim" / "gz"

CAMERA = dict(width=1280, height=854, hfov_deg=52.0, rate_hz=4)

MODEL_CONFIG = """<?xml version="1.0"?>
<model><name>{name}</name><version>1.0</version><sdf version="1.9">model.sdf</sdf>
<description>{desc}</description></model>
"""


def tile_visual(name: str, x: float, y: float, size: float, tex: str) -> str:
    return f"""
      <visual name="{name}">
        <pose>{x:.2f} {y:.2f} 0 0 0 0</pose>
        <geometry><plane><normal>0 0 1</normal><size>{size} {size}</size></plane></geometry>
        <material>
          <diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>
          <pbr><metal><albedo_map>{tex}</albedo_map><roughness>1.0</roughness><metalness>0.0</metalness></metal></pbr>
        </material>
      </visual>"""


def zephyr_cam_sdf(zephyr_sdf: str) -> str:
    """Take ardupilot_gazebo's zephyr_with_ardupilot model.sdf and add a nadir camera on the wing."""
    w, h, fov = CAMERA["width"], CAMERA["height"], math.radians(CAMERA["hfov_deg"])
    cam = f"""
    <link name="nadir_cam_link">
      <pose>0 0 -0.05 0 0 0</pose>
      <inertial><mass>0.01</mass><inertia><ixx>1e-6</ixx><iyy>1e-6</iyy><izz>1e-6</izz></inertia></inertial>
      <sensor name="nadir_cam" type="camera">
        <!-- camera looks along +X of its frame; pitch +90 deg points it down (-Z of a z-up model frame) -->
        <pose>0 0 0 0 1.5707963 0</pose>
        <always_on>1</always_on>
        <update_rate>{CAMERA['rate_hz']}</update_rate>
        <topic>nadir_cam</topic>
        <camera>
          <horizontal_fov>{fov:.6f}</horizontal_fov>
          <image><width>{w}</width><height>{h}</height><format>R8G8B8</format></image>
          <clip><near>1</near><far>5000</far></clip>
        </camera>
      </sensor>
      <sensor name="nadir_boxes" type="boundingbox_camera">
        <!-- same pose/intrinsics as nadir_cam: 2D boxes of every entity with a Label plugin (auto-labelling) -->
        <pose>0 0 0 0 1.5707963 0</pose>
        <always_on>1</always_on>
        <update_rate>{CAMERA['rate_hz']}</update_rate>
        <topic>nadir_boxes</topic>
        <camera>
          <box_type>2d</box_type>
          <horizontal_fov>{fov:.6f}</horizontal_fov>
          <image><width>{w}</width><height>{h}</height></image>
          <clip><near>1</near><far>5000</far></clip>
        </camera>
      </sensor>
    </link>
    <joint name="nadir_cam_joint" type="fixed">
      <parent>zephyr::wing</parent>
      <child>nadir_cam_link</child>
    </joint>
"""
    sdf = zephyr_sdf.replace('<model name="zephyr_with_ardupilot">', '<model name="zephyr_cam">', 1)
    i = sdf.rfind("</model>")
    return sdf[:i] + cam + sdf[i:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="03")
    ap.add_argument("--name", default="visloc03")
    ap.add_argument("--lat0", type=float, default=32.3229)
    ap.add_argument("--lon0", type=float, default=119.8530)
    ap.add_argument("--n", type=int, default=4, help="tiles per side")
    ap.add_argument("--tile-m", type=float, default=1000.0)
    ap.add_argument("--tile-px", type=int, default=2048)
    ap.add_argument("--zephyr-sdf", required=True, help="path to ardupilot_gazebo/models/zephyr_with_ardupilot/model.sdf")
    ap.add_argument("--elevation", type=float, default=5.0)
    ap.add_argument("--texture", default=None,
                    help="optional north-up ground image covering exactly the world extent (e.g. the drone-photo "
                         "orthomosaic from sim/make_mosaic.py); default: the satellite map itself")
    args = ap.parse_args()

    fl = VisLocFlight(args.flight)
    m_lat, m_lon = meters_per_degree(args.lat0)
    model = f"satmap_{args.name}"
    mdir = OUT / "models" / model
    (mdir / "textures").mkdir(parents=True, exist_ok=True)

    visuals = []
    half = args.n * args.tile_m / 2
    texture = None
    if args.texture:
        texture = cv2.cvtColor(cv2.imread(args.texture), cv2.COLOR_BGR2RGB)
        tpx = texture.shape[0] / args.n                     # texture px per tile
    for r in range(args.n):          # rows north -> south
        for c in range(args.n):      # cols west -> east
            x = -half + (c + 0.5) * args.tile_m     # East
            y = half - (r + 0.5) * args.tile_m      # North
            lat, lon = args.lat0 + y / m_lat, args.lon0 + x / m_lon
            if texture is None:
                img = fl.sat.crop(lat, lon, args.tile_m, args.tile_m / args.tile_px)
            else:
                img = texture[int(r * tpx):int((r + 1) * tpx), int(c * tpx):int((c + 1) * tpx)]
                img = cv2.resize(img, (args.tile_px, args.tile_px), interpolation=cv2.INTER_LINEAR)
            name = f"tile_r{r}_c{c}"
            cv2.imwrite(str(mdir / "textures" / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            visuals.append(tile_visual(name, x, y, args.tile_m, f"model://{model}/textures/{name}.png"))
            print("tile", name, f"E={x:.0f} N={y:.0f}")

    (mdir / "model.config").write_text(MODEL_CONFIG.format(name=model, desc="satellite ground"))
    (mdir / "model.sdf").write_text(f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{model}">
    <static>true</static>
    <link name="ground">
      <collision name="collision">
        <geometry><plane><normal>0 0 1</normal><size>{2 * half} {2 * half}</size></plane></geometry>
      </collision>{''.join(visuals)}
    </link>
  </model>
</sdf>
""")

    zdir = OUT / "models" / "zephyr_cam"
    zdir.mkdir(parents=True, exist_ok=True)
    (zdir / "model.config").write_text(MODEL_CONFIG.format(name="zephyr_cam", desc="Zephyr + nadir camera"))
    (zdir / "model.sdf").write_text(zephyr_cam_sdf(Path(args.zephyr_sdf).read_text()))

    wdir = OUT / "worlds"
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / f"{args.name}.sdf").write_text(f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="{args.name}">
    <physics name="1ms" type="ignore">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <scene>
      <ambient>0.9 0.9 0.9</ambient>
      <background>0.75 0.82 0.9</background>
      <shadows>false</shadows>
    </scene>
    <spherical_coordinates>
      <latitude_deg>{args.lat0}</latitude_deg>
      <longitude_deg>{args.lon0}</longitude_deg>
      <elevation>{args.elevation}</elevation>
      <heading_deg>0</heading_deg>
      <surface_model>EARTH_WGS84</surface_model>
    </spherical_coordinates>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 1000 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>-0.3 0.2 -0.9</direction>
    </light>
    <include><uri>model://{model}</uri></include>
    <include>
      <uri>model://zephyr_cam</uri>
      <pose degrees="true">0 0 0.422 -90 0 180</pose>
    </include>
  </world>
</sdf>
""")
    w, fov = CAMERA["width"], math.radians(CAMERA["hfov_deg"])
    (wdir / f"{args.name}.json").write_text(json.dumps(dict(
        flight=args.flight, lat0=args.lat0, lon0=args.lon0, elevation=args.elevation,
        extent_m=2 * half, camera=dict(CAMERA, focal_px=w / 2 / math.tan(fov / 2)),
        texture=args.texture or "satellite",
    ), indent=2))
    print("world written to", wdir / f"{args.name}.sdf")


if __name__ == "__main__":
    main()
