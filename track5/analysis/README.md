# analysis/ — R-side calibration and decision-curve analysis

`calibration_dca.R` produces the calibration, ECE, decision-curve and
threshold-drift figures from one long predictions CSV. It is the only R in the
project; everything upstream of it is Python.

## Running it

```
python scripts/metrics_json_to_csv.py \
    --metrics-dir runs/<run>/metrics \
    --out runs/<run>/predictions.csv \
    --threshold-out runs/<run>/threshold.txt

Rscript analysis/calibration_dca.R runs/<run>/predictions.csv
```

Outputs land in `out/` relative to the working directory:
`metrics_by_condition.csv`, `cost_ratio_thresholds.csv`, `fig1_calibration.png`,
`fig1b_ece.png`, `fig2_dca.png`, `fig3_threshold_drift.png`.

`ggplot2` is used when present and there is a base-R fallback for every figure,
so it runs on a bare R install.

## Two properties that are easy to break

**The threshold is read, never hardcoded.** `TAU` comes from `threshold.txt`,
which `metrics_json_to_csv.py --threshold-out` writes next to the CSV. If that
file is absent the script stops instead of falling back to a default. A stale
default is the worst failure available here: all three figures render, the exit
code is 0, and every number is computed at the wrong operating point. Pass an
explicit path as the second argument if the file lives elsewhere.

**The file must stay pure ASCII with no BOM.** R fails on line 1 of a file that
starts with a UTF-8 BOM, and the symptom looks like an empty file rather than an
encoding error. It also carries CRLF line endings because it is edited on
Windows. Check both before committing a change:

```
python -c "d=open('analysis/calibration_dca.R','rb').read(); \
print('BOM', d.startswith(b'\xef\xbb\xbf'), 'ASCII', all(b<0x80 for b in d))"
```

## Condition names

`COND_ORDER` must equal `transforms.eval_atoms.EVAL_15`, in that order — it is
the factor level set, and a name that does not match produces NA rows and a
silently smaller n rather than an error. `FOCUS` is the five worst-case
conditions shown in the calibration plot; `DCA_CONDS` is the two used for the
decision curve.
