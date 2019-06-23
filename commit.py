#!/usr/bin/env python
import json
import os

import pygit2
from pygit2 import (
    GIT_FILEMODE_BLOB as filemode,
    GIT_FILEMODE_BLOB_EXECUTABLE as filemode_exe,
)


repo = pygit2.Repository(".")

ok_status = frozenset({pygit2.GIT_STATUS_CURRENT, pygit2.GIT_STATUS_IGNORED})
for k, v in repo.status().items():
    if v not in ok_status:
        raise ValueError(f"Bad git status for {k}")

if repo.head.shorthand != "source":
    raise ValueError("expected to be on the `source` branch")

treebuilder = repo.TreeBuilder(repo[repo.head.target].tree)

treebuilder.remove("build.py")
treebuilder.remove("commit.py")
treebuilder.remove("Makefile")
treebuilder.remove("README-setup.md")
treebuilder.insert(
    "README.md", repo.create_blob_fromworkdir("README-setup.md"), filemode
)

for fn in ["practical.ipynb"]:
    blob = repo.create_blob_fromworkdir(fn)
    mode = filemode_exe if os.access(fn, os.X_OK) else filemode
    treebuilder.insert(fn, blob, mode)

# munge solutions.ipynb to include the readme
with open("solutions.ipynb") as f:
    nb = json.load(f)
cell = next(c for c in nb["cells"] if c["cell_type"] == "code")
assert "display(Markdown(" in cell["source"][-1]
cell["cell_type"] = "markdown"
cell["source"] = cell["outputs"][0]["data"]["text/markdown"]
del cell["outputs"], cell["execution_count"]
solutions_blob = repo.create_blob(json.dumps(nb, indent=1).encode("utf-8"))
treebuilder.insert("solutions.ipynb", solutions_blob, filemode)

# munge the gitignore
with open(".gitignore") as f:
    while f.readline().strip() != "# END OF SOURCE IGNORES":
        continue
    rest = f.read()
    assert len(rest) > 0
    gitignore_id = repo.create_blob(rest.encode("utf-8"))
treebuilder.insert(".gitignore", gitignore_id, filemode)

tree_id = treebuilder.write()

parent_ids = [repo.head.target]
built = repo.lookup_branch("built")
if built is not None:
    parent_ids.insert(0, built.target)  # apparently wants things backwards...

old_msg = repo[repo.head.target].message
line = old_msg.strip().split("\n")[0]
msg = f"Built for {str(repo.head.target)[:8]} ({line})"

commit = repo.create_commit(
    "refs/heads/built",
    repo.default_signature,
    repo.default_signature,
    msg,
    tree_id,
    parent_ids,
)
print(f"Created commit {str(commit)[:8]} on branch 'built'")
