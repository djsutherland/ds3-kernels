#!/usr/bin/env python

import json
import re


with open("solutions.ipynb") as f:
    nb = json.load(f)

skip_cell = re.compile(r"\s*(<!--|#)\s*SOLUTION CELL")
skip_line = re.compile(r".*(<!--|#)\s* SOLUTION(-->)?\s*$")

new_cells = []
for cell in nb['cells']:
    if any(skip_cell.match(line) for line in cell['source'][:3]):
        continue

    new = cell.copy()
    if "execution_count" in new:
        new["execution_count"] = None
    if "outputs" in new:
        new["outputs"] = []
    new_cells.append(new)

new_nb = nb.copy()
new_nb['cells'] = new_cells

with open("practical.ipynb", "w") as f:
    json.dump(new_nb, f, indent=1)
