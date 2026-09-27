"""Build <team_name>_submission.zip with the required layout (run from student_resource/).

python make_package.py <team_name> [--from runs/v4]

--from takes the two output files from a run directory and stores them as output/*.tsv inside the zip, without
modifying output/ (default: output/).  The validator (with --check-ids) must PASS or nothing is written.
"""
import argparse
import os
import subprocess
import sys
import zipfile

ap = argparse.ArgumentParser()
ap.add_argument("team", nargs="?", default="team")
ap.add_argument("--from", dest="src_dir", default="output")
a = ap.parse_args()
match = os.path.join(a.src_dir, "matching_results.tsv")
cand = os.path.join(a.src_dir, "candidate_pairs.tsv")
check = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", match, "--candidate", cand,
                        "--test-dir", "dataset/test", "--check-ids"], capture_output=True, text=True)
print(check.stdout.strip())
if check.returncode != 0 or "PASS" not in check.stdout:
    sys.exit("validator did not PASS - not packaging")

members = {"output/matching_results.tsv": match, "output/candidate_pairs.tsv": cand,
           "Documentation_template.md": "Documentation_template.md"}
code = "code/business_entity_resolution"
for f in ("README.md", "requirements.txt", "run_all.sh"):
    members[f"{code}/{f}"] = f"{code}/{f}"
src = f"{code}/src"
for f in sorted(os.listdir(src)):
    if f.endswith(".py"):
        members[f"{src}/{f}"] = f"{src}/{f}"
out = f"{a.team}_submission.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for arc, path in members.items():
        z.write(path, arc)
print(f"wrote {out} (outputs from {a.src_dir}):")
for arc in members:
    print("  ", arc)
