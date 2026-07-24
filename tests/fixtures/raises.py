# Fixture: crashes with a non-zero exit -> run_experiment must report exec_failed
# and carry the ValueError text through into the message.
raise ValueError("boom_from_raises")
