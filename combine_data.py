"""
combine_data.py
---------------
Merge the driving_log.csv files that each team member recorded into a single
combined CSV, without moving or copying any images.

Expected layout (one sub-folder per recording session / per teammate):

    data/
        member_a/
            driving_log.csv
            IMG/ ...
        member_b/
            driving_log.csv
            IMG/ ...
        member_c/
            driving_log.csv
            IMG/ ...

Run:
    python combine_data.py --data-dir data --out data/driving_log_combined.csv

The combined CSV keeps the standard columns. The image path columns are
rewritten to paths RELATIVE to --data-dir (e.g. member_a/IMG/xyz.jpg) so the
whole data/ folder stays portable: zip it, move it, and training still finds
every image.
"""

import argparse
import os

import pandas as pd

from utils import LOG_COLUMNS, load_log, _basename


def find_sessions(data_dir):
    """Return every sub-folder that contains a driving_log.csv."""
    sessions = []
    for entry in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, entry)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, 'driving_log.csv')):
            sessions.append(sub)
    # also handle the case where data_dir itself is a single session
    if not sessions and os.path.isfile(os.path.join(data_dir, 'driving_log.csv')):
        sessions.append(data_dir)
    return sessions


def rewrite_path(raw, session_dir, data_dir):
    """
    Rewrite one stored image path to '<session>/IMG/<filename>' relative to
    data_dir. Verifies the image actually exists.
    """
    name = _basename(raw)
    abs_img = os.path.join(session_dir, 'IMG', name)
    rel = os.path.relpath(abs_img, data_dir).replace(os.sep, '/')
    return rel, os.path.isfile(abs_img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data',
                    help='folder containing one sub-folder per recording session')
    ap.add_argument('--out', default=None,
                    help='output CSV path (default: <data-dir>/driving_log_combined.csv)')
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_path = args.out or os.path.join(data_dir, 'driving_log_combined.csv')

    sessions = find_sessions(data_dir)
    if not sessions:
        raise SystemExit(f"No driving_log.csv found in any sub-folder of {data_dir}")

    print(f"Found {len(sessions)} session(s):")
    for s in sessions:
        print("   ", os.path.relpath(s, data_dir))

    frames, total_missing = [], 0
    for session in sessions:
        df = load_log(os.path.join(session, 'driving_log.csv'))
        for col in ('center', 'left', 'right'):
            if col in df.columns:
                results = df[col].apply(lambda r: rewrite_path(r, session, data_dir))
                df[col] = [r[0] for r in results]
                if col == 'center':
                    total_missing += sum(0 if r[1] else 1 for r in results)
        frames.append(df[LOG_COLUMNS])
        print(f"   + {os.path.relpath(session, data_dir)}: {len(df)} rows")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)

    print(f"\nCombined {len(combined)} total rows -> {out_path}")
    if total_missing:
        print(f"WARNING: {total_missing} center image(s) referenced in the logs "
              f"were not found in their IMG/ folders. Check that each session's "
              f"IMG folder was copied alongside its driving_log.csv.")


if __name__ == '__main__':
    main()
