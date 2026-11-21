#!/usr/bin/env bash
# Install pinned source with the user-requested Torch/torchvision metadata override.
# No model weights or runtime dependencies are downloaded by this script.
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_upstream="${1:-${task_root}/third_party/Isaac-GR00T}"
task_python="${VLA_TTT_PYTHON:-python3}"
task_commit="3b49067943eb1ab2eb2203b93d461c4390c78da3"

"${task_python}" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Pinned GR00T requires Python 3.12"'
if [[ -e "${task_upstream}" ]]; then
    if [[ ! -d "${task_upstream}/.git" ]]; then
        echo "Refusing to overwrite a non-git directory: ${task_upstream}" >&2
        exit 1
    fi
    task_head="$(git -C "${task_upstream}" rev-parse HEAD)"
    if [[ "${task_head}" != "${task_commit}" ]]; then
        echo "Existing checkout is at another commit; refusing to replace it: ${task_upstream}" >&2
        exit 1
    fi
else
    mkdir -p -- "${task_upstream}"
    git -C "${task_upstream}" init
    git -C "${task_upstream}" remote add origin https://github.com/NVIDIA/Isaac-GR00T.git
    git -C "${task_upstream}" fetch --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git "${task_commit}"
    git -C "${task_upstream}" checkout --detach "${task_commit}"
fi

# Accept a pristine checkout or exactly this already-applied patch. Never reset,
# stash, or overwrite unrelated user changes. Only project dependency entries are
# changed; optional flash-attention build metadata and all model source stay as-is.
"${task_python}" - "${task_upstream}" "${task_commit}" <<'PY'
from pathlib import Path
import re
import subprocess
import sys
import tomllib

checkout = Path(sys.argv[1])
commit = sys.argv[2]

def git(*args):
    return subprocess.check_output(["git", "-C", str(checkout), *args])

original = git("show", f"{commit}:pyproject.toml")
text = original.decode("utf-8")
updated = text
for package, before, after in (("torch", "2.9.0", "2.9.1"), ("torchvision", "0.24.0", "0.24.1")):
    pattern = rf'(?m)^(\s*"{package}==){re.escape(before)}(",\s*)$'
    updated, count = re.subn(pattern, lambda match: match[1] + after + match[2], updated)
    if count != 1:
        raise SystemExit(f"Expected exactly one standalone upstream {package} dependency; found {count}")

before_deps = tomllib.loads(text)["project"]["dependencies"]
after_deps = tomllib.loads(updated)["project"]["dependencies"]
expected = [dependency.replace("torch==2.9.0", "torch==2.9.1").replace(
    "torchvision==0.24.0", "torchvision==0.24.1") for dependency in before_deps]
if after_deps != expected:
    raise SystemExit("Dependency override changed unexpected project metadata")
if git("diff", "--cached", "--name-only").strip():
    raise SystemExit("Existing upstream checkout has staged changes; preserving them")
changed = set(git("diff", "--name-only").decode().splitlines())
if changed - {"pyproject.toml"}:
    raise SystemExit(f"Existing upstream checkout has unrelated changes: {sorted(changed)}")
if git("ls-files", "--others", "--exclude-standard").strip():
    raise SystemExit("Existing upstream checkout has untracked files; preserving them")
target = checkout / "pyproject.toml"
patched = updated.encode("utf-8")
if target.is_symlink():
    raise SystemExit("pyproject.toml is a symlink; refusing to replace user-managed metadata")
if target.read_bytes() not in (original, patched):
    raise SystemExit("pyproject.toml contains other user changes; refusing to overwrite it")
if target.read_bytes() != patched:
    target.write_bytes(patched)
print("Applied dependency override: torch 2.9.1 / torchvision 0.24.1 (user-selected, not upstream validated)")
PY
"${task_python}" -m pip install --no-deps -e "${task_upstream}"
echo "Installed GR00T source ${task_commit} with the documented dependency override; install the shared VLA-TTT runtime separately."
