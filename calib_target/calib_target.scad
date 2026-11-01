// Calibration target: 150 mm hollow sphere on a stem + ISO 9409-1-50-4-M6 flange adapter (UFACTORY 850).
// Same geometry as make_calib_target.py. Render one part at a time: PART = "flange" | "lower" | "upper" | "foam_hub".
PART = "flange";

ball_d = 150; wall = 1.6; ball_center_z = 180;
lip_h = 6; lip_clear = 0.25;
stem_r = 11; stem_insert = 25; collar_h = 20; socket_clear = 0.3; boss_r = stem_r + 5;
lock = "insert";                    // "insert": M4 heat-set pocket Ø5.6x6 + Ø4.3 through; "tap": Ø3.4 pilot for M4
insert_d = 5.6; insert_h = 6; screw_d = 4.3; tap_d = 3.4;
dimple_d = 8; dimple_h = 3;
foam_plug_d = 35; foam_plug_l = 40; foam_disc_d = 60; foam_disc_t = 3;
flange_d = 63; flange_t = 8; pcd = 50; bolt_d = 6.6; cbore_d = 11; cbore_t = 4; pin_d = 6.2;
spigot_d = 31.2; spigot_h = 3;      // set spigot_h = 0 if the robot flange has no Ø31.5 H7 recess
fillet_r = 18; fillet_h = 10;
$fn = 128;

R = ball_d / 2;
stem_top = ball_center_z - R + stem_insert;
screw_z = ball_center_z - R - collar_h / 2;     // flange frame

module lock_cut(z) {                            // radial along +X at height z (part frame)
  if (lock == "insert") translate([boss_r - insert_h, 0, z]) rotate([0, 90, 0]) cylinder(d = insert_d, h = insert_h + 0.5, $fn = 32);
  translate([-1, 0, z]) rotate([0, 90, 0]) cylinder(d = (lock == "insert") ? screw_d : tap_d, h = boss_r + 2, $fn = 32);
}

module flange_stem() {
  difference() {
    union() {
      cylinder(d = flange_d, h = flange_t);
      if (spigot_h > 0) translate([0, 0, -spigot_h]) cylinder(d = spigot_d, h = spigot_h + 0.01);
      translate([0, 0, flange_t - 0.01]) cylinder(r1 = fillet_r, r2 = stem_r, h = fillet_h);
      translate([0, 0, flange_t - 0.01]) cylinder(r = stem_r, h = stem_top - flange_t + 0.01);
    }
    for (k = [0:3]) rotate([0, 0, 45 + 90 * k]) translate([pcd / 2, 0, -spigot_h - 1]) {
      cylinder(d = bolt_d, h = flange_t + spigot_h + 2, $fn = 48);
      translate([0, 0, flange_t + spigot_h + 1 - cbore_t]) cylinder(d = cbore_d, h = cbore_t + 1, $fn = 48);
    }
    translate([pcd / 2, 0, -spigot_h - 1]) cylinder(d = pin_d, h = flange_t + spigot_h + 2, $fn = 48);   // Ø6 dowel at +X (ISO 9409-1)
    translate([stem_r + 1, 0, screw_z]) rotate([0, -90, 0]) cylinder(d1 = dimple_d + 2, d2 = 1, h = dimple_h + 1, $fn = 32);  // screw dimple
    translate([0, 0, stem_top - 1.5]) difference() { cylinder(r = stem_r + 5, h = 3); cylinder(r1 = stem_r - 1.5, r2 = stem_r + 5, h = 3.01); }  // entry chamfer
  }
}

module shell() { difference() { sphere(r = R); sphere(r = R - wall); } }

module lower() {                                // part frame: sphere centre at origin
  difference() {
    union() {
      intersection() { shell(); translate([0, 0, -R - 1]) cylinder(r = R + 2, h = R + 1); }
      difference() { cylinder(r = R - wall - lip_clear, h = lip_h); translate([0, 0, -0.5]) cylinder(r = R - wall - lip_clear - 2, h = lip_h + 1); }
      translate([0, 0, -R - collar_h]) cylinder(r = boss_r, h = collar_h + stem_insert + 3);   // collar: outside -> inside
    }
    translate([0, 0, -R - collar_h - 1]) cylinder(r = stem_r + socket_clear, h = collar_h + stem_insert + 1);
    lock_cut(-R - collar_h / 2);
  }
}

module upper() { intersection() { shell(); cylinder(r = R + 2, h = R + 1); } }

module foam_hub() {                             // z=0 = surface of the foam ball
  difference() {
    union() {
      cylinder(d = foam_disc_d, h = foam_disc_t);
      translate([0, 0, foam_disc_t - 0.01]) cylinder(d = foam_plug_d - 0.4, h = foam_plug_l);
      translate([0, 0, -collar_h]) cylinder(r = boss_r, h = collar_h + 0.01);
    }
    translate([0, 0, -collar_h - 1]) cylinder(r = stem_r + socket_clear, h = collar_h + foam_disc_t + stem_insert + 1);
    lock_cut(-collar_h / 2);
  }
}

if (PART == "flange") flange_stem();
if (PART == "lower") lower();
if (PART == "upper") upper();
if (PART == "foam_hub") foam_hub();
