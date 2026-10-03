#!/usr/bin/env bash
# Source from a launch script; initialize an existing Conda installation.
if [[ -n "${RGBA_CONDA_ROOT:-}" ]]; then
  source "$RGBA_CONDA_ROOT/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
  eval "$("$CONDA_EXE" shell.bash hook)"
else
  printf '%s\n' 'Initialize Conda or set RGBA_CONDA_ROOT to your Conda installation.' >&2
  return 1
fi
conda activate anime
# An interactive worker can reset PATH while retaining CONDA_PREFIX.
export PATH="$CONDA_PREFIX/bin:$PATH"
