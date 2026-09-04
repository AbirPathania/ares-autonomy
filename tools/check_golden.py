"""Check that the committed golden file still matches the document.

Regenerates every table in memory and compares it to
ares_math/tests/golden_values.yaml. Any change to a formula, a transcribed
document value, or the shape of a table fails here.

The comparison is numeric at 1e-12 relative tolerance rather than a byte
diff, because tan and sqrt differ in the last bit between libm
implementations, and a golden file that only holds on the machine that wrote
it is not a golden file. 1e-12 is three orders tighter than the 1e-9 the
build spec asks the tests to hold, so nothing real slips through.

Run:  python tools/check_golden.py
"""

import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_golden import OUT, build_document  # noqa: E402

TOL = 1e-12


def compare(got, want, path, problems):
    if isinstance(want, dict):
        if not isinstance(got, dict):
            problems.append(f"{path}: expected a mapping, file has {type(got).__name__}")
            return
        for key in set(want) | set(got):
            if key not in want:
                problems.append(f"{path}.{key}: in the file, not in the generator")
            elif key not in got:
                problems.append(f"{path}.{key}: in the generator, not in the file")
            else:
                compare(got[key], want[key], f"{path}.{key}", problems)
    elif isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            problems.append(
                f"{path}: {len(want)} rows generated, "
                f"{len(got) if isinstance(got, list) else 'not a list'} in the file"
            )
            return
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            compare(g, w, f"{path}[{i}]", problems)
    elif isinstance(want, float) and isinstance(got, (int, float)):
        if not math.isclose(got, want, rel_tol=TOL, abs_tol=1e-15):
            problems.append(f"{path}: file has {got!r}, generator gives {want!r}")
    elif got != want:
        problems.append(f"{path}: file has {got!r}, generator gives {want!r}")


def main():
    if not OUT.exists():
        sys.exit(f"{OUT} does not exist. Run tools/generate_golden.py.")
    with OUT.open() as fh:
        committed = yaml.safe_load(fh)

    problems = []
    compare(committed, build_document(), "golden", problems)

    if problems:
        print("golden_values.yaml no longer matches the document:\n")
        for p in problems[:40]:
            print(f"  {p}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        print(
            "\nIf this is a deliberate change, run tools/generate_golden.py "
            "and commit the result."
        )
        sys.exit(1)

    rows = sum(len(v) for v in committed.values() if isinstance(v, list))
    print(f"golden_values.yaml matches the document, {rows} rows checked")


if __name__ == "__main__":
    main()
