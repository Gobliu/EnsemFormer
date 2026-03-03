"""
Prepare CycPeptMPDB-4D dataset:
  1. Renumber MODEL blocks in all trajectory PDBs to sequential 1–100.
  2. Parse GROMACS clustering logs, map ns times → 1-based frame indices,
     and add Water_RepFrame / Hexane_RepFrame columns to the CSV.

Usage:
    python scripts/prepare_dataset.py --data_root /path/to/CycPeptMPDB_4D \
                                      --csv data/CycPeptMPDB-4D.csv
"""

import argparse
import re
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants derived from dataset inspection
# ---------------------------------------------------------------------------
# MD frames: steps 10_000_000 → 24_850_000, spacing 150_000 (= 0.3 ns @ 2 fs/step)
# Simulation time of frame i (1-based): t_ns = 20.0 + (i - 1) * 0.3
T0_NS = 20.0       # time of frame 1
DT_NS = 0.3        # time spacing between frames
N_FRAMES = 100


def time_to_frame(t_ns: float) -> int:
    """Convert GROMACS log simulation time (ns) to 1-based PDB frame index."""
    return min(max(round((t_ns - T0_NS) / DT_NS) + 1, 1), N_FRAMES)


# ---------------------------------------------------------------------------
# Task 1: renumber MODEL blocks
# ---------------------------------------------------------------------------

def renumber_traj_pdb(pdb_path: Path) -> str:
    """Rewrite MODEL lines in a trajectory PDB to use sequential 1-based numbering.
    Returns 'ok' or an error string."""
    try:
        lines = pdb_path.read_text()
        frame_idx = 0
        out_lines = []
        for line in lines.splitlines(keepends=True):
            if line.startswith("MODEL"):
                frame_idx += 1
                out_lines.append(f"MODEL{frame_idx:>9}\n")
            else:
                out_lines.append(line)
        if frame_idx != N_FRAMES:
            return f"WARN {pdb_path.name}: found {frame_idx} MODEL blocks (expected {N_FRAMES})"
        pdb_path.write_text("".join(out_lines))
        return "ok"
    except Exception as e:
        return f"ERROR {pdb_path}: {e}"


def renumber_all_trajs(data_root: Path, n_workers: int = 8):
    water_trajs = sorted((data_root / "Water" / "Trajectories").glob("*.pdb"))
    hexane_trajs = sorted((data_root / "Hexane" / "Trajectories").glob("*.pdb"))
    all_trajs = water_trajs + hexane_trajs

    print(f"Renumbering {len(all_trajs)} trajectory PDBs with {n_workers} workers...")
    errors = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(renumber_traj_pdb, p): p for p in all_trajs}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Renumber"):
            result = f.result()
            if result != "ok":
                errors.append(result)

    if errors:
        print(f"\n{len(errors)} issues:")
        for e in errors:
            print(" ", e)
    else:
        print("All trajectory PDBs renumbered successfully.")


# ---------------------------------------------------------------------------
# Task 2: parse logs → representative frame index
# ---------------------------------------------------------------------------

def parse_log_rep_frame(log_path: Path) -> int | None:
    """Parse a GROMACS clustering log and return the 1-based frame index of
    the representative conformer (middle structure of cluster 1)."""
    text = log_path.read_text()

    # Find the cluster table. The first data line after the header gives cluster 1.
    # Format: "  1 |  42  0.094 |     32 .078 | ..."
    # We need the 'middle' value (first number after the second '|')
    table_pattern = re.compile(
        r"cl\.\s*\|\s*#st\s+rmsd\s*\|\s*middle\s+rmsd\s*\|.*?\n"  # header
        r"\s*1\s*\|\s*\d+\s+[\d.]+\s*\|\s*([\d.]+)",               # cluster 1 line
        re.DOTALL,
    )
    m = table_pattern.search(text)
    if not m:
        return None
    middle_ns = float(m.group(1))
    return time_to_frame(middle_ns)


def build_rep_frame_columns(data_root: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Add Water_RepFrame and Hexane_RepFrame columns to df."""
    water_log_dir = data_root / "Water" / "Logs"
    hexane_log_dir = data_root / "Hexane" / "Logs"

    water_frames = {}
    hexane_frames = {}

    print("Parsing Water logs...")
    for log in tqdm(sorted(water_log_dir.glob("*.log")), desc="Water logs"):
        # Extract ID from filename: {YEAR}_{Author}_{ID}_H2O.log
        parts = log.stem.split("_")
        pid = int(parts[2])
        frame = parse_log_rep_frame(log)
        water_frames[pid] = frame

    print("Parsing Hexane logs...")
    for log in tqdm(sorted(hexane_log_dir.glob("*.log")), desc="Hexane logs"):
        parts = log.stem.split("_")
        pid = int(parts[2])
        frame = parse_log_rep_frame(log)
        hexane_frames[pid] = frame

    df = df.copy()
    df["Water_RepFrame"] = df["CycPeptMPDB_ID"].map(water_frames).astype("Int64")
    df["Hexane_RepFrame"] = df["CycPeptMPDB_ID"].map(hexane_frames).astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare CycPeptMPDB-4D dataset")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/home/liuw/GitHub/Data/CycPeptMPDB_4D"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "CycPeptMPDB-4D.csv",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Parallel workers for PDB renumbering",
    )
    parser.add_argument(
        "--skip_renumber", action="store_true",
        help="Skip trajectory PDB renumbering (if already done)",
    )
    args = parser.parse_args()

    if not args.skip_renumber:
        renumber_all_trajs(args.data_root, n_workers=args.workers)

    print("\nBuilding representative frame columns...")
    df = pd.read_csv(args.csv)
    df = build_rep_frame_columns(args.data_root, df)

    missing_w = df["Water_RepFrame"].isna().sum()
    missing_h = df["Hexane_RepFrame"].isna().sum()
    if missing_w or missing_h:
        print(f"WARNING: {missing_w} missing Water_RepFrame, {missing_h} missing Hexane_RepFrame")

    df.to_csv(args.csv, index=False)
    print(f"CSV updated: {args.csv}")
    print(df[["CycPeptMPDB_ID", "Source", "Water_RepFrame", "Hexane_RepFrame"]].head(10).to_string())


if __name__ == "__main__":
    main()
