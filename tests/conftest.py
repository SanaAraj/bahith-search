import os

# Disable the startup embedding warmup before any app modules are imported, so
# the test suite never triggers a model download.
os.environ.setdefault("WARMUP_ON_STARTUP", "false")
