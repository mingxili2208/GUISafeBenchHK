#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

base = Path('/home/hp/STF/SafeBenchHK-zh-simulate-tag/safebench/scenario/scenario_data/14C_2')
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
