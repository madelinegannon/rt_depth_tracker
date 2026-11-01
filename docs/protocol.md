# Wire protocol

OSC over UDP, one message per datagram, no bundles, plain scalar arguments. Everything is loopback by
default (`127.0.0.1`); the tracker binds `127.0.0.1` only. Ports: see the table in the README. Every port
and address is a constant in `src/rt_depth_tracker/protocol.py`.

Frame: WORLD is Z-up, metres, origin on the floor. The sensor frame is x right, y down, z forward
(optical axis). A calibration file maps sensor -> world.

## The sensor contract (in)

This is the plug point for a new sensor. A capture process is anything that sends these two messages;
nothing downstream imports a camera SDK. `rt_depth_tracker.capture` is the one shipped (Orbbec Femto Mega
over the network, via pyorbbecsdk2).

| address | arguments | meaning |
|---|---|---|
| `/depth` | `cam_id:int seq:int width:int height:int stride:int device_ts:double chunk_idx:int n_chunks:int blob` | one chunk of one depth frame. `width`, `height` are the full-resolution size; the blob holds the image sub-sampled every `stride`-th pixel in both axes (`height // stride` rows of `width // stride` little-endian uint16, millimetres, 0 = no return), cut into `n_chunks` pieces of at most 8000 bytes in order. `seq` counts frames per camera from 1; `device_ts` is the camera clock in seconds. |
| `/intrinsics` | `cam_id:int width:int height:int fx:float fy:float cx:float cy:float k1 k2 k3 k4 k5 k6 p1 p2:float` | the full-resolution pinhole + Brown-Conrady model (OpenCV convention: radial k1..k6 as a rational, tangential p1, p2). Sent about once a second; the tracker keeps the latest and writes it to `config/calib/intrinsics_<cam>.json` for the offline tools. A rectified sensor sends eight zeros. |
| `/hb` | `name:string json:string` | once a second to port 9000: a name and a JSON dict of metrics (fps, valid pixels, phase, `busy_s` lease before a blocking SDK call). Optional; only a supervisor reads it. |

Rules the tracker relies on: pixel (u, v) of the strided image is full-resolution pixel (u·stride, v·stride);
depth is distance along the optical axis, not along the ray; a frame's chunks arrive within a second; a
restarted sender may count `seq` from 1 again. Two cameras are paired by arrival: a step runs when every
live camera has delivered a new frame, or 34 ms after the first one did.

## The tracker's output (out)

Sent to every `host:port` in `--out` (the consumer first, then mirrors such as the viewer).

| address | arguments | meaning |
|---|---|---|
| `/tracker/target` | `x:float y:float z:float hint:string track_id:int score:float` | every frame (camera rate). The one point to attend to and how: `idle` (nobody worth attention; `x y z` = the configured `idle_target`), `watch` (someone, mildly), `engage` (someone, fully; `x y z` near their head), `reach` (a hand over the rail or toward the base; `x y z` = that hand), `sleep` (the presence gate says nobody is around; see `sleep_gate`). `track_id` -1 when idle or asleep; `score` the interest score in [0, 1]. |
| `/tracker/tracks` | `n:int blob` | 10 Hz. `n` rows of 8 little-endian float32: `[id, top_x, top_y, top_z, height, score, reach, flags]` for every confirmed track. `top` is the head point, `height` the smoothed stature, `reach` how far the body extends toward the base, `flags` bit 0 = over the rail, bit 1 = the current target. `n` may be 0 (the blob is then one byte). |
| `/tracker/zone` | `idx:int name:string occupied:int reaching:int track_id:int dwell_s:float leaning:int` | every frame, one per configured zone, in config order; never sent without zones. `occupied` 1 when someone stands at it (nearest zone within `zone_near_m`), `reaching` 1 when someone's hand is over it in the hand band, `leaning` 1 when someone's head is in the head band over its near edge for at least `lean_min_s`; `track_id` the reaching person, else the leaning one, else the occupying one, else -1; `dwell_s` seconds the zone has been occupied, reached or leaned over without a gap longer than `zone_hold_s`. |
| `/hb` | `name:string json:string` | once a second: `targets`, `hint`, `presence`, per-camera receive rates, calibration and background state, `q_age` (None without a robot), zone states, step time. |

Ids are stable while a person is tracked and never reused in a run. Hints hold through short dropouts
(a track coasts for `lost_misses` frames).

## The robot's measured pose (in, optional)

| address | arguments | meaning |
|---|---|---|
| `/robot/q` | `q1 q2 ... qn:float` | the robot's measured joint angles in radians, in URDF joint order, from whatever talks to the robot. Only listened to (port 7011) when the config has a `robot` block; then the arm is masked out of the cloud at that pose. A feed older than `robot_q_stale_s` (or none at all) widens every sphere to `robot_stale_margin`, so an arm whose pose is unknown cannot leak through as a visitor. |

The tracker never sends anything to this port and never sends a joint command anywhere.

## Control (in)

To the tracker's control port (7012), from `rt_depth_tracker.ctl`, `calib_tool`, or your own supervisor:

| address | arguments | effect |
|---|---|---|
| `/tracker/learn_bg` | `[frames:int]` | relearn the background from the next N frames (default `bg_learn_frames`), then save `config/calib/bg_<cam>.npz` |
| `/tracker/record` | `tag:string` or `0` | start a tagged clip (`recordings/<tag>_<time>_<n>.npz`, 30 s chunks) / stop |
| `/tracker/reload_calib` | | re-read `sensor_*.json`, `bg_*.npz`, `intrinsics_*.json` (also happens within a second of a file change) |
| `/tracker/set` | `key:string value:float` | live-tune `near_pref`, `kid_pref`, `kid_h` or `adult_h`; written to the config once the value settles |
| `/tracker/quit` | | clean stop, exit 0 |
| `/capture/quit` | | to a capture's `--ctrl` port: stop the stream, release the camera, exit 0 |

## Receiving

```python
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import numpy as np

def on_target(addr, x, y, z, hint, track_id, score):
    print(hint, round(x, 2), round(y, 2), round(z, 2), track_id, round(score, 2))

def on_tracks(addr, n, blob):
    rows = np.frombuffer(blob, dtype="<f4", count=n * 8).reshape(n, 8)   # id top_x top_y top_z height score reach flags

def on_zone(addr, idx, name, occupied, reaching, track_id, dwell_s, leaning):
    print(name, "LEAN" if leaning else "REACH" if reaching else "at" if occupied else "-", track_id)

d = Dispatcher()
d.map("/tracker/target", on_target)
d.map("/tracker/tracks", on_tracks)
d.map("/tracker/zone", on_zone)
BlockingOSCUDPServer(("127.0.0.1", 7001), d).serve_forever()
```

Or import the package: `rt_depth_tracker.protocol.unpack(datagram)` returns a dataclass per message, and
`rt_depth_tracker.run.TargetRx(port)` is a threaded receiver of all three.
