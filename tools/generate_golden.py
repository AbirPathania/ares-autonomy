"""Generate ares_math/tests/golden_values.yaml.

Build spec section 11, mechanism 1: every table in section 03 becomes
golden_values.yaml, asserted by pytest at 1e-9 relative tolerance.

This script is deliberately independent of ares_math. It writes the formulas
out inline, from the document, and it hard-codes the rounded values that are
printed in the PDF. It then asserts that its own full-precision computation
rounds to what the document prints, and only then writes the YAML.

So there are three things that must agree before a golden file exists:

    1. the printed table in the build spec PDF, transcribed by hand below
    2. this script's inline implementation of the formula
    3. ares_math, checked against the YAML by pytest

If the YAML were generated from ares_math, the test suite would be asserting
that the code equals itself. It is generated here instead so that the
document is the outer check.

Run:  python tools/generate_golden.py
"""

import math
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parents[1] / "ares_math" / "tests" / "golden_values.yaml"

# Section 03 working case, and the constants the document computed with.
F_WORKING = 640.0
W_PX = 1280
HFOV = 90.0
VFOV = 59.0
SIGMA_NOMINAL = 0.25
SIGMA_PESSIMISTIC = 0.50
MAX_DISPARITY = 192
TRENCH_W = 0.5
TRENCH_D = 0.3
SAFETY_FACTOR = 1.5
LATENCY = 0.30
DECEL = 1.0


def rnd(x, places):
    """Round half away from zero, which is what a document table does."""
    if x == math.inf:
        return math.inf
    factor = 10.0**places
    return math.floor(abs(x) * factor + 0.5) / factor * (1.0 if x >= 0 else -1.0)


# ---------------------------------------------------------------------------
# Document errata.
#
# Cells where the build spec's printed value disagrees with the build spec's
# own formula. Each one is recorded here rather than silently accommodated,
# because "silent divergence between the document and the code" is exactly
# what section 11 exists to catch. The formula wins; the printed cell is
# noted. Remove an entry only when the document is reissued with it fixed,
# and expect the test suite to tell you loudly when that happens.
# ---------------------------------------------------------------------------
ERRATA = {
    "d_near h=0.95 tilt=40": {
        "printed": 0.35,
        "computed": 0.3551904455105644,
        "correct_rounding": 0.36,
        "note": (
            "Section 03.4 blind zone table, mast row, 40 degree column. "
            "h/tan(tilt + VFOV/2) with h=0.95 and VFOV=59 gives 0.35519, "
            "which rounds to 0.36 at two places. Every other cell in the "
            "table rounds correctly, so this is a single typo and not a "
            "different formula. It does not affect any conclusion: the mast "
            "is specified at 20 degrees of tilt, not 40."
        ),
    },
}

_errata_hit = set()


def check(computed, printed, places, label):
    got = rnd(computed, places)
    if got != printed:
        entry = ERRATA.get(label)
        if entry is None:
            raise AssertionError(
                f"{label}: document prints {printed}, formula gives {got} "
                f"(full precision {computed!r}). If the document is wrong, add "
                f"an ERRATA entry saying so. Do not adjust the formula."
            )
        if entry["printed"] != printed or rnd(entry["computed"], 10) != rnd(
            computed, 10
        ):
            raise AssertionError(
                f"{label}: an errata entry exists but no longer matches. "
                f"Recorded printed={entry['printed']} computed={entry['computed']}, "
                f"now printed={printed} computed={computed!r}."
            )
        _errata_hit.add(label)
    return float(computed)


# ---------------------------------------------------------------------------
# 03.1 Focal length in pixels.  f_px = (W_px / 2) / tan(HFOV / 2)
# ---------------------------------------------------------------------------
FOCAL_PRINTED = [
    (1280, 90.0, 640.0, "Working case used throughout this document"),
    (1280, 110.0, 448.1, "Wide lens, worse depth precision for the same baseline"),
    (1920, 90.0, 960.0, "Same optics, more pixels, better disparity precision"),
    (1920, 72.0, 1321.3, "Narrow, for a long-range unit"),
]


def focal_table():
    rows = []
    for width, hfov, printed, note in FOCAL_PRINTED:
        f = (width / 2.0) / math.tan(math.radians(hfov) / 2.0)
        rows.append(
            {
                "width_px": width,
                "hfov_deg": hfov,
                "expected": check(f, printed, 1, f"focal {width}/{hfov}"),
                "document": printed,
                "note": note,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 03.2 Stereo depth error.  dZ = Z^2 * d_disp / (f * B)
# ---------------------------------------------------------------------------
DEPTH_RANGES = [2, 4, 6, 8, 10, 20]
BASELINES = [0.06, 0.12, 0.20, 0.30]
DEPTH_PRINTED = {
    2: [0.026, 0.013, 0.008, 0.005],
    4: [0.104, 0.052, 0.031, 0.021],
    6: [0.234, 0.117, 0.070, 0.047],
    8: [0.417, 0.208, 0.125, 0.083],
    10: [0.651, 0.326, 0.195, 0.130],
    20: [2.604, 1.302, 0.781, 0.521],
}


def depth_error_table():
    rows = []
    for z in DEPTH_RANGES:
        for i, b in enumerate(BASELINES):
            dz = z**2 * SIGMA_NOMINAL / (F_WORKING * b)
            rows.append(
                {
                    "Z": float(z),
                    "B": b,
                    "f": F_WORKING,
                    "d_disp": SIGMA_NOMINAL,
                    "expected": check(dz, DEPTH_PRINTED[z][i], 3, f"dZ Z={z} B={b}"),
                    "document": DEPTH_PRINTED[z][i],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# 03.2 Baseline selection.
#   range at target error  Z = sqrt(err * f * B / d_disp)
#   minimum range          Z_min = f * B / d_max
# ---------------------------------------------------------------------------
BASELINE_PRINTED = {
    0.06: (3.92, 2.77, 0.20),
    0.12: (5.54, 3.92, 0.40),
    0.20: (7.16, 5.06, 0.67),
    0.30: (8.76, 6.20, 1.00),
}


def baseline_table():
    rows = []
    for b in BASELINES:
        nominal_p, pessimistic_p, minimum_p = BASELINE_PRINTED[b]
        nominal = math.sqrt(0.10 * F_WORKING * b / SIGMA_NOMINAL)
        pessimistic = math.sqrt(0.10 * F_WORKING * b / SIGMA_PESSIMISTIC)
        minimum = F_WORKING * b / MAX_DISPARITY
        rows.append(
            {
                "B": b,
                "f": F_WORKING,
                "target_error": 0.10,
                "max_disparity_px": MAX_DISPARITY,
                "range_at_10cm": check(nominal, nominal_p, 2, f"range@10cm B={b}"),
                "range_at_10cm_document": nominal_p,
                "range_at_10cm_pessimistic": check(
                    pessimistic, pessimistic_p, 2, f"range@10cm pess B={b}"
                ),
                "range_at_10cm_pessimistic_document": pessimistic_p,
                "min_range": check(minimum, minimum_p, 2, f"min range B={b}"),
                "min_range_document": minimum_p,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 03.3 Stopping distance.  D_stop = v * t_latency + v^2 / (2a)
#
# The document's speed column is km/h and its v column is that divided by 3.6
# and printed to three places. The full precision value is what reproduces the
# total column, so that is what is stored here.
# ---------------------------------------------------------------------------
STOPPING_PRINTED = [
    # speed_kmh, a, v, reaction, braking, total, times_1_5
    (2, 1.0, 0.556, 0.167, 0.154, 0.321, 0.481),
    (4, 1.0, 1.111, 0.333, 0.617, 0.951, 1.426),
    (4, 2.0, 1.111, 0.333, 0.309, 0.642, 0.963),
    (8, 1.0, 2.222, 0.667, 2.469, 3.136, 4.704),
]


def stopping_table():
    rows = []
    for kmh, a, v_p, react_p, brake_p, total_p, x15_p in STOPPING_PRINTED:
        v = kmh / 3.6
        react = v * LATENCY
        brake = v * v / (2.0 * a)
        total = react + brake
        rows.append(
            {
                "speed_kmh": kmh,
                "a": a,
                "t_latency": LATENCY,
                "safety_factor": SAFETY_FACTOR,
                "v": check(v, v_p, 3, f"v {kmh} km/h"),
                "v_document": v_p,
                "reaction": check(react, react_p, 3, f"reaction {kmh} a={a}"),
                "reaction_document": react_p,
                "braking": check(brake, brake_p, 3, f"braking {kmh} a={a}"),
                "braking_document": brake_p,
                "total": check(total, total_p, 3, f"total {kmh} a={a}"),
                "total_document": total_p,
                "total_x_safety": check(
                    total * SAFETY_FACTOR, x15_p, 3, f"x1.5 {kmh} a={a}"
                ),
                "total_x_safety_document": x15_p,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 03.4 Blind zone.  d_near = h / tan(tilt + VFOV/2), VFOV = 59 degrees
# ---------------------------------------------------------------------------
BLIND_HEIGHTS = [0.35, 0.50, 0.95, 1.25]
BLIND_TILTS = [0, 10, 20, 30, 40]
BLIND_PRINTED = {
    0.35: [0.62, 0.42, 0.30, 0.21, 0.13],
    0.50: [0.88, 0.61, 0.43, 0.29, 0.19],
    0.95: [1.68, 1.15, 0.81, 0.56, 0.35],
    1.25: [2.21, 1.52, 1.07, 0.74, 0.47],
}


def blind_zone_table():
    rows = []
    for h in BLIND_HEIGHTS:
        for i, tilt in enumerate(BLIND_TILTS):
            d = h / math.tan(math.radians(tilt + VFOV / 2.0))
            rows.append(
                {
                    "h": h,
                    "tilt_deg": float(tilt),
                    "vfov_deg": VFOV,
                    "expected": check(
                        d, BLIND_PRINTED[h][i], 2, f"d_near h={h} tilt={tilt}"
                    ),
                    "document": BLIND_PRINTED[h][i],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# 03.5 Pixels on target.  n_px = H * f / Z
# ---------------------------------------------------------------------------
PIXEL_OBJECTS = [
    ("standing_person", 1.70, [218, 109, 54, 27]),
    ("crouching_person", 1.00, [128, 64, 32, 16]),
    ("rock_030", 0.30, [38, 19, 10, 5]),
    ("rock_015", 0.15, [19, 10, 5, 2]),
]
PIXEL_RANGES = [5, 10, 20, 40]


def pixels_table():
    rows = []
    for name, height, printed in PIXEL_OBJECTS:
        for i, z in enumerate(PIXEL_RANGES):
            n = height * F_WORKING / z
            rows.append(
                {
                    "object": name,
                    "H": height,
                    "f": F_WORKING,
                    "Z": float(z),
                    "expected": check(n, printed[i], 0, f"px {name} @{z}"),
                    "document": printed[i],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Section 02 Negative obstacle range.  R_max = w * h / d
# ---------------------------------------------------------------------------
NEG_HEIGHTS = [0.35, 0.50, 0.75, 1.00, 1.25]
NEG_PRINTED = {0.35: 0.58, 0.50: 0.83, 0.75: 1.25, 1.00: 1.67, 1.25: 2.08}
NEG_VERDICT = {
    # speed_kmh: verdict per height in NEG_HEIGHTS order
    2: ["OK", "OK", "OK", "OK", "OK"],
    4: ["FAIL", "FAIL", "FAIL", "OK", "OK"],
    8: ["FAIL", "FAIL", "FAIL", "FAIL", "FAIL"],
}
STOP_X15_PRINTED = {2: 0.48, 4: 1.43, 8: 4.70}


def negative_obstacle_table():
    rows = []
    for h in NEG_HEIGHTS:
        r_max = TRENCH_W * h / TRENCH_D
        row = {
            "h": h,
            "w": TRENCH_W,
            "d": TRENCH_D,
            "expected": check(r_max, NEG_PRINTED[h], 2, f"R_max h={h}"),
            "document": NEG_PRINTED[h],
            "verdicts": {},
        }
        for kmh, verdicts in NEG_VERDICT.items():
            v = kmh / 3.6
            stop = (v * LATENCY + v * v / (2.0 * DECEL)) * SAFETY_FACTOR
            got = "OK" if r_max >= stop else "FAIL"
            want = verdicts[NEG_HEIGHTS.index(h)]
            if got != want:
                raise AssertionError(
                    f"verdict h={h} at {kmh} km/h: document says {want}, got {got} "
                    f"(R_max {r_max:.4f} vs stop {stop:.4f})"
                )
            row["verdicts"][kmh] = got
        rows.append(row)
    return rows


def stopping_x_safety_table():
    rows = []
    for kmh, printed in STOP_X15_PRINTED.items():
        v = kmh / 3.6
        stop = (v * LATENCY + v * v / (2.0 * DECEL)) * SAFETY_FACTOR
        rows.append(
            {
                "speed_kmh": kmh,
                "a": DECEL,
                "t_latency": LATENCY,
                "safety_factor": SAFETY_FACTOR,
                "expected": check(stop, printed, 2, f"stop x1.5 {kmh} km/h"),
                "document": printed,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Section 02 Mast height.  h = d * D_stop / w
# ---------------------------------------------------------------------------
MAST_PRINTED = {2: 0.29, 4: 0.86, 8: 2.82}


def mast_height_table():
    rows = []
    for kmh, printed in MAST_PRINTED.items():
        v = kmh / 3.6
        stop = (v * LATENCY + v * v / (2.0 * DECEL)) * SAFETY_FACTOR
        h = TRENCH_D * stop / TRENCH_W
        rows.append(
            {
                "speed_kmh": kmh,
                "w": TRENCH_W,
                "d": TRENCH_D,
                "stop_distance": stop,
                "expected": check(h, printed, 2, f"mast height {kmh} km/h"),
                "document": printed,
            }
        )
    return rows


def build_document():
    """Assemble every table. Each value is checked against the document."""
    doc = {
        "_README": (
            "Frozen numeric tables from the ARES Autonomy Build Spec v1, "
            "4 September 2026. Generated by tools/generate_golden.py, which "
            "computes each value from the formula in the document and asserts "
            "it rounds to the value the document prints. Do not hand edit. "
            "ares_math is tested against this file at 1e-9 relative tolerance."
        ),
        "_spec": "ARES Autonomy Build Spec v1",
        "_spec_date": "2026-09-04",
        "_hull_revision": "H-2",
        "working_case": {
            "f_px": F_WORKING,
            "width_px": W_PX,
            "hfov_deg": HFOV,
            "vfov_deg": VFOV,
            "disparity_sigma_px": SIGMA_NOMINAL,
            "disparity_sigma_px_pessimistic": SIGMA_PESSIMISTIC,
            "max_disparity_px": MAX_DISPARITY,
            "trench_width": TRENCH_W,
            "trench_depth": TRENCH_D,
            "latency": LATENCY,
            "decel": DECEL,
            "safety_factor": SAFETY_FACTOR,
        },
        "focal_table": focal_table(),
        "depth_error_table": depth_error_table(),
        "baseline_table": baseline_table(),
        "stopping_table": stopping_table(),
        "stopping_x_safety_table": stopping_x_safety_table(),
        "blind_zone_table": blind_zone_table(),
        "pixels_table": pixels_table(),
        "negative_obstacle_table": negative_obstacle_table(),
        "mast_height_table": mast_height_table(),
    }
    missed = set(ERRATA) - _errata_hit
    if missed:
        raise AssertionError(
            f"errata recorded for cells that now agree with the document: "
            f"{sorted(missed)}. If the spec was reissued, delete the entry."
        )
    doc["_errata"] = ERRATA
    return doc


def main():
    doc = build_document()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False, width=100)
    counted = sum(len(v) for v in doc.values() if isinstance(v, list))
    print(f"wrote {OUT} with {counted} rows, every one checked against the document")
    print(f"{len(ERRATA)} document errata recorded, see the _errata key")


if __name__ == "__main__":
    main()
