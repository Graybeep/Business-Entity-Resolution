"""Build <team_name>_submission.zip with the required layout (run from student_resource/).

python make_package.py <team_name>
"""
import os
import subprocess
import sys
import zipfile

team = sys.argv[1] if len(sys.argv) > 1 else "team"
check = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", "output/matching_results.tsv",
                        "--candidate", "output/candidate_pairs.tsv", "--test-dir", "dataset/test"],
                       capture_output=True, text=True)
print(check.stdout.strip())
if check.returncode != 0:
    sys.exit("validator did not PASS - not packaging")

members = ["output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
           "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt"]
src = "code/business_entity_resolution/src"
members += [os.path.join(src, f).replace("\\", "/") for f in sorted(os.listdir(src)) if f.endswith(".py")]
out = f"{team}_submission.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for m in members:
        z.write(m, m)
print(f"wrote {out}:")
for m in members:
    print("  ", m)
