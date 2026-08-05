# Sandbox for agent-written training code.
#
# Built once; agents never install packages. Containers run with `--network none`,
# so a package missing from uv.lock is missing for good — the agent cannot pip
# install its way out. That is why the dependency set is deliberately generous.
#
# Threat model is accidental damage, not an adversarial attacker. Plain Docker
# with no network, a read-only data mount, dropped capabilities, and a wall-clock
# kill is enough; gVisor/Firecracker would be solving a problem we do not have.
FROM python:3.12-slim

# libgomp1 is the OpenMP runtime. LightGBM, XGBoost and CatBoost all dlopen it
# and `python:3.12-slim` does not ship it — without this every boosted model
# dies at import with "libgomp.so.1: cannot open shared object file".
#
# That failure is invisible from inside the sandbox: there is no network, so the
# agent cannot apt-get its way out, and the traceback points at the import line
# rather than at the image. The debug action would retry forever against code
# that was never wrong. Fix it here, once.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv, straight from its published image — no curl, no install script, no network
# dependency at build time beyond the two base images.
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /usr/local/bin/

# The virtualenv must NOT live under /workspace: that path is bind-mounted at
# run time, and the mount would shadow it. /opt/venv is outside every mount.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

# The dependency layer is by far the most expensive, so it goes first and is
# keyed only on the two files that define it.
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Import every library at BUILD time. A missing shared object (the libgomp case
# above) or a broken wheel otherwise surfaces as a node failure at 3am, looking
# for all the world like the agent wrote bad code. Failing the build instead
# turns a whole class of phantom bugs into a red build.
RUN python -c "\
import numpy, pandas, scipy, sklearn, lightgbm, xgboost, catboost, polars, pyarrow; \
from sklearn.model_selection import StratifiedGroupKFold; \
from lightgbm import LGBMClassifier; \
from xgboost import XGBClassifier; \
from catboost import CatBoostClassifier; \
print('sandbox imports OK:', \
      'numpy', numpy.__version__, '| pandas', pandas.__version__, \
      '| sklearn', sklearn.__version__, '| lightgbm', lightgbm.__version__, \
      '| xgboost', xgboost.__version__, '| catboost', catboost.__version__)"

# A non-root default. The executor additionally passes `--user <host uid>:<gid>`
# so writes to the bind-mounted /workspace land with host ownership; this USER
# is the floor if someone runs the image by hand.
RUN useradd --create-home --uid 1000 runner

# Under `--user`, HOME may point somewhere unwritable. Libraries that want a
# cache directory (matplotlib, numba, joblib) then fail with a permission error
# that reads exactly like a bug in the agent's code, and the debug action would
# chase it forever. Point them all at the one directory guaranteed writable.
ENV HOME=/workspace \
    MPLCONFIGDIR=/workspace/.cache/matplotlib \
    XDG_CACHE_HOME=/workspace/.cache \
    NUMBA_CACHE_DIR=/workspace/.cache/numba \
    JOBLIB_TEMP_FOLDER=/workspace/.cache/joblib \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

WORKDIR /workspace
USER runner

CMD ["python", "train.py"]
