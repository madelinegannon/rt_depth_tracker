# rt_depth_tracker

A realtime people tracker on depth images. Depth frames come in over OSC from one or two cameras; the tracker
subtracts a learned background, clusters the rest on the floor plane, tracks each person, scores how
interesting they are, and sends one attention target plus per-track rows (and, optionally, per-zone states)
back out over OSC at the camera rate.

**This package never commands a robot.** It contains no motion code and no robot SDK. It was written as the
perception half of robot installations, so it can mask a robot's own arm out of the depth cloud (given a URDF
and a feed of measured joint angles) and the target it emits is shaped for a behaviour app that moves a
robot, but whatever consumes `/tracker/target` is a separate program with its own safety gates.

Two things plug in, both optional in the sense that you pick what fits:

- **Sensor.** The tracker only ever sees `/depth` (a strided uint16 depth image, mm) and `/intrinsics`
  (a Brown-Conrady camera model) on loopback UDP. `rt_depth_tracker.capture` sends them for an Orbbec
  Femto Mega over the network; any other sensor is a new process that sends the same two messages
  (`docs/protocol.md`). A rectified sensor sends zero distortion coefficients. Nothing downstream imports
  the camera SDK.
- **Calibration.** Each camera needs its pose in the world (Z-up, metres, origin on the floor). Three ways,
  each a chain of `calib_tool` verbs (`docs/calibration.md`): floor plane + yaw from a floor mark (a tape
  measure, a person standing still), a sphere target on a robot flange (needs the robot's measured joint
  angles on `/robot/q`; you move the arm, the tool never does), or a rectangle of known size in view (a
  table top, for overhead rigs). The viewer's Calibration panel nudges the result live.

Copyright 2024-2026 ATONATON, LLC. Apache-2.0; see `LICENSE` and `NOTICE`.

## Install

```
python -m venv .venv
.venv/bin/pip install -e ".[viewer]"        # tracker + the 3D viewer and the matplotlib panels
.venv/bin/pip install -e ".[capture]"       # + pyorbbecsdk2, only on the machine with the cameras
```

Python 3.10+. Windows: `.venv\Scripts\python` in place of `.venv/bin/python`.

## Run

```
cp config/tracker.example.json config/tracker.json      # edit: camera positions, robot (optional), zones (optional)
python -m rt_depth_tracker.capture --ip <camera ip> --cam-id 0      # one process per camera -> /depth on 7010
python -m rt_depth_tracker.capture --ip <camera ip> --cam-id 1 --out 127.0.0.1:7010,127.0.0.1:7013
python -m rt_depth_tracker.run --config config/tracker.json         # /tracker/target -> 127.0.0.1:7001
python -m rt_depth_tracker.calib_tool learn-bg                      # empty room: the background
python -m rt_depth_tracker.viewer --live                            # beside it, on the mirror ports
```

Then calibrate (`docs/calibration.md`). Without a calibration the tracker runs and sees nobody; without a
background it sees nobody either (blind beats "everything is foreground").

| port | who -> whom | what |
|---|---|---|
| 7010 | capture -> tracker | `/depth`, `/intrinsics` |
| 7013 | capture -> viewer | the same, mirrored (`--out 127.0.0.1:7010,127.0.0.1:7013`) |
| 7001 | tracker -> your consumer | `/tracker/target`, `/tracker/tracks`, `/tracker/zone` |
| 7016 | tracker -> viewer | the same, mirrored (`--out 127.0.0.1:7001,127.0.0.1:7016`) |
| 7011 | robot side -> tracker | `/robot/q`, only listened to with a robot in the config |
| 7012 | ctl / calib_tool -> tracker | control verbs |
| 9000 | every process -> a supervisor | `/hb` once a second |

Console scripts for the same modules: `rt-tracker`, `rt-capture`, `rt-calib`, `rt-ctl`, `rt-replay`,
`rt-viewer`, `rt-viz`.

## Config

`config/tracker.json`, live-reloaded; a gitignored `tracker.local.json` beside it overrides keys per machine.
Every key and its default is in `src/rt_depth_tracker/config.py`. The ones that describe your space:

- `cams`: per camera, a measured position and a point it looks at. The seed for every calibration method.
- `mount_z`, `base_xy`, `base_yaw_deg`: where the robot base stands, if there is one. The crop box and the
  cluster grid are centred on `base_xy` whether or not there is a robot.
- `robot`: `{"urdf": ..., "robot_description": ...}` (paths relative to the config file). Set it and the
  tracker masks the arm at the pose it hears on `/robot/q`; leave it empty and nothing is masked. The
  description YAML lists collision spheres per link; `tests/fixtures/` holds a pair for the UFACTORY 850.
- `rail_radius`, `rail_top_z`: a barrier around the base; a hand over it is a "reach". 0 = none.
- `zones`: rectangles with a top height (a table, a plinth). Who stands at one, who reaches over it,
  who leans in: `/tracker/zone`, sent only when zones are configured.
- `table`: a rectangle of known size and height for the table calibration step.
- `marks`: named floor points for the yaw step, drawn by the viewers.
- `wall_x`, `pedestal_radius`: drawn and (the pedestal) masked; `null` / 0 = none.

## Outputs

`/tracker/target x y z hint track_id score` at the camera rate: where to look and how (`idle`, `watch`,
`engage`, `reach`, `sleep`), `/tracker/tracks` at 10 Hz with every confirmed person, `/tracker/zone` per
configured zone. Formats and a receiver example: `docs/protocol.md`.

## Tools

- `rt_depth_tracker.ctl record <tag> | record off | learn-bg | reload-calib | set <key> <value>`
- `rt_depth_tracker.replay recordings/<tag>_*.npz --loop`: the identical pipeline on a clip, any machine
- `rt_depth_tracker.viewer --clip ... | --live [--screenshot out.png]`: the orbit viewer with every layer a
  checkbox, the Calibration panel and the Zones panel
- `rt_depth_tracker.viz --clip ... | --live | --coverage`: matplotlib panels and a floor-coverage map
- `calib_target/`: a printable 150 mm sphere target with an ISO 9409-1-50 flange adapter, for the sphere method

## Tests

```
.venv/bin/python tests/run_all.py          # 13 suites, hand-rolled check() scripts, synthetic scenes, no hardware
```

## Layout

```
src/rt_depth_tracker/   capture, protocol, background, intrinsics, pacing, segment, tracks, zones, pipeline,
                        interest, presence, robot_model, run, ctrl_rx, record, replay, ctl, calib, calib_tool,
                        rgb, viewer, viz, synth
tests/                  one script per suite + run_all.py; fixtures/ the UF850 URDF + sphere YAML
docs/                   protocol.md, calibration.md
config/                 tracker.example.json (calib/ and *.local.json are yours and gitignored)
calib_target/           SCAD, generator script and STEP files of the sphere target
```
