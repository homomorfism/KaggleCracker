# Fixture: exits 0 and prints a number for something else, but never the required
# 'CV_SCORE: <float>' line -> bad_input, and the stray 0.8814 must not leak back
# as the score.
print("loading data")
print("training fold 1")
print("training fold 2")
print("mean auc = 0.8814")
print("done")
