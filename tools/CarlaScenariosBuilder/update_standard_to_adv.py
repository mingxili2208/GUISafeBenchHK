#!/usr/bin/env python3
import json
import shutil
import sys
from pathlib import Path

# Use repo root to locate scenario data, falling back to cwd
_repo_root = Path(__file__).resolve().parents[2]
base = _repo_root / "safebench" / "scenario" / "scenario_data" / "14C_2"
if not base.exists():
    print(f"Default base not found: {base}", file=sys.stderr)
    print("Pass the base directory as the first argument, e.g.:", file=sys.stderr)
    print("  python update_standard_to_adv.py /path/to/scenario_data/14C_2", file=sys.stderr)
    sys.exit(1)
files = ['standard_scenario_01.json', 'standard_scenario_02.json']

for fname in files:
    p = base / fname
    if not p.exists():
        print(f"File not found: {p}")
        continue
    # make backup
    backup = p.with_suffix(p.suffix + '.bak')
    shutil.copy2(p, backup)
    print(f"Backup created: {backup}")
    data = json.loads(p.read_text())
    changed = 0
    for entry in data:
        # only modify if scenario_folder is 'standard' or parameters is None
        if entry.get('scenario_folder') == 'standard':
            entry['scenario_folder'] = 'adv_init_state'
            changed += 1
        if entry.get('parameters') is None:
            entry['parameters'] = ["genetic_algorithm", True]
            changed += 1
    p.write_text(json.dumps(data, indent=2))
    print(f"Updated {p}: made {changed} field changes")
