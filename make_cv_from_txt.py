from pathlib import Path
import re
import pandas as pd

txt_path = Path("reports/cv_rf_vs_lr.txt")
csv_path = Path("reports/cv_rf_vs_lr.csv")

raw = txt_path.read_text(encoding="utf-8", errors="ignore")
lines = [ln.rstrip("\n") for ln in raw.splitlines()]

# Find the line that contains the column headings (the one with RandomForest / LogReg)
start = 0
for i, ln in enumerate(lines):
    if "RandomForest" in ln and "LogReg" in ln:
        start = i + 1
        break

rows = []
rx = re.compile(r"""
    ^\s*
    (?P<metric>[A-Za-z0-9_\.]+)
    \s+(?P<rf_mean>[-+]?\d*\.\d+|\d+)
    \s+(?P<rf_std>[-+]?\d*\.\d+|\d+)
    \s+(?P<lr_mean>[-+]?\d*\.\d+|\d+)
    \s+(?P<lr_std>[-+]?\d*\.\d+|\d+)
    \s*$
""", re.VERBOSE)

for ln in lines[start:]:
    m = rx.match(ln)
    if m:
        rows.append(m.groupdict())

df = pd.DataFrame(rows, columns=["metric","rf_mean","rf_std","lr_mean","lr_std"])
df.to_csv(csv_path, index=False, encoding="utf-8")
print("Wrote", csv_path, "with columns:", list(df.columns), "and", len(df), "rows.")
