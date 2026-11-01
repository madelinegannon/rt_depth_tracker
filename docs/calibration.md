# Camera calibration: how it works and how to do it

Everything here runs from the repo root with the venv (`.venv/bin/python`, on Windows `.venv\Scripts\python`).

## 1. What "calibration" means here

Each depth camera delivers a depth image: for every pixel, the distance to the first surface, in mm. The
tracker turns that into 3-D points in the **world frame** (Z-up, metres, origin on the floor; a robot base, if
there is one, stands at `base_xy`, `mount_z` above the floor). To do that it needs, per camera:

| what | file | where it comes from |
|---|---|---|
| **intrinsics** (fx, fy, cx, cy, distortion) | sent by the capture process at 1 Hz, stored in every clip, mirrored to `config/calib/intrinsics_<cam>.json` | the camera's factory calibration; never edited |
| **extrinsics** `T_world_sensor` (4x4: where the camera is and which way it looks) | `config/calib/sensor_a.json` (cam 0), `sensor_b.json` (cam 1) | **this document** |
| **background** (per-pixel distance of the empty room) | `config/calib/bg_<cam>.npz` | `learn-bg` with nobody in the room |

`T_world_sensor` has six degrees of freedom: position (x, y, z) and orientation (roll, pitch, yaw). A 1 deg
orientation error moves a point 3.5 m away by 6 cm; a 15 deg error moves it by 90 cm and a person's head
reads 2.3 m high. The world crop, the zones, the robot self-mask and the two-camera merge all assume the
extrinsics are right, so **every symptom below is a calibration symptom before it is a tracker symptom**.

The calibration files carry a `sha` and a `meta.method` chain (for example `seed+plane+yaw`). The tracker
logs the sha and picks up new files within a second of a change (or on `ctl reload-calib`). Every step keeps
the previous file as `sensor_<x>.prev.json`; `calib_tool undo` puts it back.

## 2. Three methods

Pick the one your setup allows. They chain: each step reads whatever calibration is on disk and fixes some of
the six degrees of freedom.

| method | chain | needs | fixes | accuracy |
|---|---|---|---|---|
| **floor + mark** | `seed -> learn-bg -> plane -> yaw` | a tape measure, the floor in view, one measured floor mark, a person to stand on it | all six (position from the tape, roll/pitch/height from the floor, yaw from the mark) | ~5 cm in 15 minutes |
| **sphere** | `seed -> learn-bg -> sphere [-> refine] -> validate` | a robot with a `robot` block in the config, its measured joint angles on `/robot/q`, the printed ball target on its flange, the cameras' colour streams | all six, ~5 mm / 5 mrad | the real calibration |
| **table** | `seed -> learn-bg -> plane -> table` | a rectangle of known size and height in view (a table top, a plinth), `table` in the config | all six (roll/pitch/height from the floor, yaw + xy from the rectangle) | ~2 cm |

And on top of any of them: `nudge --yaw --dx --dy`, or the viewer's Calibration panel sliders, for a manual
world-frame correction you can see.

### What each step measures

| step | measures | fixes | cannot fix | needs |
|---|---|---|---|---|
| `aim` | nothing: the real FOV and the measured camera positions | tells you where to point each lens (a floor point F per camera) so both cameras cover the zone | anything after the cameras are set | measured camera positions |
| `seed` | nothing: builds a look-at pose from `cams[].pos` / `aim` in the config | all six DOF roughly (enough for the floor and a person to be recognisable) | anything precisely | a measured camera position; an aim point |
| `learn-bg` | median distance of every pixel over N frames of the empty room | nothing in the pose, but every later step needs it | a person present during the learn becomes a permanent ghost | **empty room**, robot parked |
| `plane` | the floor as a plane through the background points | roll, pitch, height (rotates about the camera position) | yaw, lateral position | seed + background, floor visible |
| `yaw` | bearing from the camera to a person standing still on a **measured floor mark** | yaw (rotates about the camera's vertical axis); checks the camera-to-mark range as a consistency test | camera position errors: they show up as a range mismatch and the step refuses | seed + plane + background, a mark, a clip |
| `table` | the table top as a plane, its outline as a rectangle | yaw and xy (rotates/shifts the world so the rectangle sits centred on the origin, long side along x) | roll/pitch/height (run `plane` first); the 180 deg ambiguity of a rectangle | seed + plane + background, `table.size_xy` and `table.top_z` in the config |
| `sphere` | a ball on the robot flange at 10-16 arm poses, found in colour, fitted in depth, ground truth from forward kinematics at the measured joint angles | all six DOF to ~5 mm / 5 mrad, camera position included | nothing; it is the real calibration | the printed ball target, `/robot/q`, ~20 min |
| `refine` | ICP of the robot's own silhouette against its sphere model | small residuals | gross errors | a seeded pose, the robot visible, `/robot/q` |
| `validate` | leave-one-out residual of the sphere sweep, robot-mask leftovers at the current arm pose | reports only | | after `sphere`; a `/robot/q` source |
| `nudge` | nothing: applies a yaw about the world z axis and an xy shift you type | what you tell it | | a seeded pose |

### How the maths works (one paragraph each)

**Seed.** `look_at_T(pos, aim)` builds the sensor axes in the world: z toward the aim point, x horizontal
(right in the image), y down. Position comes straight from the config. A camera looking exactly straight
down has no horizontal: then image up is world +x. Aim an overhead camera exactly straight down (`pos` and
`aim` share x and y); a nearly-vertical look-at lets a millimetre of aim offset pick the image's up direction,
and the table step can only resolve yaw within 90 deg of the seed. A camera mounted rotated by 180 deg gets
`nudge --yaw 180` before `table`.

**Background.** For each pixel, the median of its valid returns and a robust spread (MAD). A pixel is
"foreground" when its return is **closer** than the median by more than a tolerance (`bg_tol_mm`, 3 sigma or
1 % of range, whichever is larger); farther never counts (multipath, flying pixels). Pixels that rarely return
anything are "invalid": any return within `bg_invalid_max_mm` there is foreground. The file holds this ground
truth only. At runtime a surface that holds perfectly still off the learned depth for `bg_absorb_s` is healed
into a session overlay (a moved chair, a sign); the moment the learned surface shows again it un-heals. The
overlay is never saved.

**Plane.** Points the current pose puts within `--z-band` of the floor go through RANSAC; the inlier plane's
normal is rotated onto +z about the camera position (so a measured position stays where the tape put it) and
the pose is lifted so the plane sits at z = 0. Reports tilt and height before and after.

**Yaw.** A clip of a person standing still on a mark at known (x, y). Frames with enough foreground points in
the tracked volume give a centroid; runs of at least `--min-run` seconds within 15 cm of each other are
candidates. For each, the bearing camera -> centroid is compared with camera -> mark and the pose rotated
about the camera's vertical axis by the difference. The range camera -> centroid must match camera -> mark
within `--max-range-err` (10 %), or the step refuses: a wrong mark, a person off the mark, or a camera that is
not where the config says all show up here. With two cameras the pair of runs that overlap most in time wins
(a bystander seen by one camera cannot). A mark off the camera's axis is the one that catches a camera
configured on the wrong side; an on-axis mark proves nothing, by symmetry.

**Table.** Points within `--band` of `table.top_z` go through RANSAC for the top plane; the inliers' outline is
fitted with a minimum-area rectangle (1 deg sweep, 0.05 deg refinement). Its measured sides must match
`table.size_xy` within 15 % or the step refuses (wrong table, occluded top, wrong `top_z`). The world is then
rotated about z and shifted so the rectangle is centred on the origin with its long side along +x (or +y when
`size_xy[1]` is the longer).

**Sphere.** For each pose the ball centre is predicted by forward kinematics (`robot.urdf`, the measured
joint angles on `/robot/q`, `--offset` along the flange z). In colour the ball is the largest orange blob near
that prediction; its depth pixels, gated to one ball depth, are fitted with a fixed-radius sphere (Gauss-Newton
on the centre, Huber weights). Depth-only (`--no-rgb`) takes the foreground points within `--search-r` of the
prediction instead, minus the arm (robot mask) and the stem. The camera pose is the Kabsch fit of the
observed centres (sensor frame) onto the predicted ones (world); the report gives the RMS and the
leave-one-out RMS, the honest number. `--append` adds a run to the saved sweep instead of replacing it.

## 3. Marking the floor (floor + mark method)

Rule of thumb: `seed -> learn-bg -> plane -> yaw` gets two cameras to agree within ~5 cm from tape-measure
inputs in 15 minutes.

1. Measure each camera's position (lens centre, not the bracket) in the world frame and write it to
   `cams[].pos` in `config/tracker.json`, with a rough aim point.
2. Mark the floor: one point on the camera's axis of interest (for a two-camera rig facing +x: a mark at
   x = 1.6, y = 0) and one or two lateral marks (x = 1.6, y = +-1.0). Put them in `marks` so the viewers draw
   them. Masking tape and a tape measure from the origin are enough.
3. `calib_tool seed --cam 0`, `--cam 1`.
4. `calib_tool learn-bg` with the room empty (the robot parked), or `ctl learn-bg` to a running tracker.
5. `calib_tool plane --cams 0,1`.
6. `ctl record yaw_ref`, a person walks to the on-axis mark and stands still for 10 s, `ctl record off`;
   `calib_tool yaw --cams 0,1 --clip recordings/yaw_ref_*.npz --ref 1.6,0`.
7. Verify on a lateral mark: another clip, `yaw --ref 1.6,1.0 --dry-run`. A `range_ratio` off by more than a
   few percent means the camera is not where the config says (a swapped pair shows 25-30 %).
8. Look: `viewer --live`. The floor from both cameras must coincide, a person must be one box of the right
   height from either camera, and the marks must land under the person standing on them.

After any camera is touched: `plane`, then `yaw` on the axis mark, then the dry run. Ten minutes.

## 4. The sphere method (with a robot)

1. Print `calib_target/` (a 150 mm sphere on a stem, ISO 9409-1-50 flange) and bolt it to the flange; the
   ball centre is 180 mm along the flange z: `--offset 0,0,0.180`.
2. Publish the robot's measured joint angles as `/robot/q` to port 7011 and, for the viewer, 7014. This is
   the robot side's job; the tracker and the tool only listen.
3. `calib_tool probe-color --cam 0 --cam-ips <ip0>,<ip1>`: the ball must show as one solid patch in the PNG;
   adjust `--hsv-lo/--hsv-hi` if not.
4. `calib_tool sphere --cams 0,1 --poses 12 --cam-ips <ip0>,<ip1>` (stop the capture processes first: the
   tool opens the cameras itself for colour). The tool prints 12 suggested ball positions spread over both
   cameras' view; you move the arm to each by whatever means you have (a teach pendant, a jog UI, your own
   program), press Enter when it is still, the tool reads the pose and the cameras. Keep the arm out of the
   ball's line of sight to each camera. `s` skips a pose, `q` finishes early.
5. Pass: `ok: true` per camera, `loo_rms_mm` < 15, the printed `cam_pos` within ~10 cm of a tape measurement
   (the lens is not where the tape lands; 10-20 cm is normal). Then `calib_tool validate` (`loo_rms_mm` < 15;
   `robot_leftover_pts` < 30 with the arm back in the pose the background was learned in).
6. Do the marks check of section 3 anyway: a sphere calibration with the base yaw or `mount_z` wrong is
   self-consistent and still wrong for the people.

## 5. The table method (overhead rigs, no robot)

For one camera above a table, plinth or display case, looking straight down.

1. `cams[0]`: the measured height, `pos` and `aim` with the same x and y (straight down). `table`:
   `{"center_xy": [0, 0], "size_xy": [long, short], "top_z": height}`; the world origin is the table centre.
2. `seed --cam 0`, `learn-bg` with the table clear of people, `plane --cams 0`, `table --cams 0`.
3. Pass: `applied: true`, `size_ratio` within a few percent of 1, `top_tilt_deg` under 1. In the viewer the
   table box (`zones` layer) must sit on the table-top points; if it is rotated by 180 deg, `nudge --yaw 180`
   and run `table` again.
4. Zones (`zones[]`) are rectangles in this frame; the viewer's Zones panel drags them onto the real things
   and saves them.

## 6. Files and hygiene

- `config/calib/` is gitignored: back it up after every site, with the measurements.
- Never `seed` over a calibrated pair unless you mean to start over; `plane`, `yaw` and `table` chain on
  whatever is there (`meta.method` tells you the history); `undo` steps back once.
- Clips live in `recordings/` (gitignored, 30 s chunks, ~80 MB per chunk); keep the reference clips of the
  final calibration.
- A background recorded at another stride or depth mode is rejected at load, not applied.

## Visual QA

`python -m rt_depth_tracker.viewer --clip recordings/<tag>_*.npz` (or `--live` beside the running tracker: the
`/depth` mirror on 7013, a `/robot/q` mirror on 7014, the tracker's output mirror on 7016) is an orbit viewer
around the tracker pipeline: left-drag orbits, middle/right-drag pans, wheel zooms; space plays/pauses a clip,
left/right step it, `r` resets the view, `s` saves a PNG, `q` quits. `--screenshot out.png --at 10` renders
offscreen. Layers, each a checkbox: background clouds per camera (the floor from both cameras must coincide:
**sensor alignment**), tracked points per camera, points the world crop dropped (grey) and the robot mask
dropped (dark red), the robot sphere set at the current q (**sensor-to-robot calibration**, needs a robot and
a q source), a box per floor-grid component (grey rejected, white candidate: **feature extraction**), a box per
track (grey tentative, yellow confirmed, green current, red reach / over the rail: **tracking quality**), gaze
and reach points, the target as sent (magenta cross) against this viewer's own decision, trails of the last
3 s, camera frustums, the table and zones (lit when occupied or reached). The Calibration panel nudges the
world under the loaded calibration (yaw / x / y) live and saves it as `+nudge`; the Zones panel edits the zone
rectangles live and saves them to the config file.
