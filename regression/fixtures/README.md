# Regression fixtures

Each file is one recorded agent transcript (prompt, tool calls with results,
final report) replayed by a test so that a failure once observed stays caught
without calling the model.

The two transcripts from the 2026-10-15 runs on the original build were lost
with that machine. `make_fixtures.py` **reconstructs** them from the real tool
output on the regenerated golden cases and the failure text as it was recorded
in the note; the reports are therefore edited oracle reports, not the model's
original prose. Provenance is stated in each file's `provenance` field.
