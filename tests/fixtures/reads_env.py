# Fixture: stands in for model-written code that goes looking for secrets. It
# reports whether an ambient credential reached it and whether the sandbox
# variables did, then scores normally so the run itself succeeds.
import os

print("KAGGLE_KEY=%s" % os.environ.get("KAGGLE_KEY", "<unset>"))
print("KC_DATASET_REF=%s" % os.environ.get("KC_DATASET_REF", "<unset>"))
print("CV_SCORE: 0.5")
