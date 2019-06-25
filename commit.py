#!/usr/bin/env python
from glob import glob
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

index = pygit2.Index()
index.read_tree(repo[repo.head.target].tree)


def index_add(name, blob, mode=filemode):
    index.add(pygit2.IndexEntry(name, blob, mode))


index.remove("build.py")
index.remove("commit.py")
index.remove("Makefile")
index.remove("README-setup.md")
index_add("README.md", repo.create_blob_fromworkdir("README-setup.md"))

for fn in ["practical.ipynb", "data/ridge-toy.npz"] + glob("figs/*.png"):
    mode = filemode_exe if os.access(fn, os.X_OK) else filemode
    index_add(fn, repo.create_blob_fromworkdir(fn), mode)

# munge solutions-ridge.ipynb to include the readme
with open("solutions-run-ridge.ipynb") as f:
    nb = json.load(f)
cell = next(c for c in nb["cells"] if c["cell_type"] == "code")
assert "display(Markdown(" in cell["source"][-1]
cell["cell_type"] = "markdown"
cell["source"] = cell["outputs"][0]["data"]["text/markdown"]
del cell["outputs"], cell["execution_count"]
solutions_blob = repo.create_blob(json.dumps(nb, indent=1).encode("utf-8"))
index_add("solutions-ridge.ipynb", solutions_blob)

index_add(
    "solutions-testing.ipynb",
    repo.create_blob_fromworkdir("solutions-run-testing.ipynb"),
)

# munge the gitignore
with open(".gitignore") as f:
    while f.readline().strip() != "# END OF SOURCE IGNORES":
        continue
    rest = f.read()
    assert len(rest) > 0
    gitignore_id = repo.create_blob(rest.encode("utf-8"))
index_add(".gitignore", gitignore_id, filemode)

tree_id = index.write_tree(repo)

parent_ids = []
built = repo.lookup_branch("built")
if built is not None:
    parent_ids.append(built.target)

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
