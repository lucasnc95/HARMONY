#!/usr/bin/env bash
# Statistics of the paper tests (output of paper_job.pbs):
#
#   scripts/pbs/paper_stats.sh [results/paper]
#
# For every benchmark / precision / scenario / configuration, from the wall
# times of its runs (timings.txt):
#   n          successful runs (of the runs asked)
#   mean       sample mean (s)
#   std        sample standard deviation (n - 1)
#   ci95       half-width of the 95 % confidence interval of the mean:
#              t(0.975, n-1) * std / sqrt(n)   (Student's t; n = 10 -> t = 2.262)
#   ci_low/ci_high  mean -/+ ci95
#   cv %       std / mean
#   per_iter   mean / iterations (ms)
#   speedup    mean of the GPU-only run (gpu1/gpu) / mean
#
# Writes <dir>/stats.csv and prints a table.

set -euo pipefail
DIR=${1:-results/paper}
OUT="$DIR/stats.csv"

shopt -s nullglob
files=("$DIR"/*/*/*/*/timings.txt)
if [ ${#files[@]} -eq 0 ]; then
    echo "no $DIR/<bench>/<precision>/<scenario>/<config>/timings.txt found" >&2
    exit 1
fi

{
    for f in "${files[@]}"; do
        d=$(dirname "$f")
        IFS=/ read -r -a parts <<< "${d#"$DIR"/}"
        info="$d/info.txt"
        iterations=$(sed -n 's/.*iterations=\([0-9]*\).*/\1/p' "$info" 2>/dev/null | head -n 1)
        runs=$(sed -n 's/.* runs=\([0-9]*\).*/\1/p' "$info" 2>/dev/null | head -n 1)
        # one line per run: bench precision scenario config iterations runs_asked seconds
        awk -v b="${parts[0]}" -v p="${parts[1]}" -v s="${parts[2]}" -v c="${parts[3]}" \
            -v it="${iterations:-0}" -v r="${runs:-0}" 'NF { print b, p, s, c, it, r, $1 }' "$f"
        # configurations whose runs all failed still get a row
        [ -s "$f" ] || echo "${parts[0]} ${parts[1]} ${parts[2]} ${parts[3]} ${iterations:-0} ${runs:-0} nan"
    done
} | awk '
    function tq(df) {   # Student t, two-sided 95 %
        split("12.706 4.303 3.182 2.776 2.571 2.447 2.365 2.306 2.262 2.228 2.201 2.179 2.160 2.145 2.131 2.120 2.110 2.101 2.093 2.086 2.080 2.074 2.069 2.064 2.060 2.056 2.052 2.048 2.045 2.042", t, " ")
        if (df < 1) return 0
        if (df <= 30) return t[df]
        if (df <= 60) return 2.000
        if (df <= 120) return 1.980
        return 1.960
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $4
        if (!(key in seen)) { seen[key] = 1; order[++nk] = key; iters[key] = $5; asked[key] = $6 }
        if ($7 != "nan") { n[key]++; sum[key] += $7; sq[key] += $7 * $7 }
    }
    END {
        print "benchmark,precision,scenario,config,n,runs,mean_s,std_s,ci95_s,ci_low_s,ci_high_s,cv_pct,per_iter_ms,speedup_vs_gpu1"
        for (k = 1; k <= nk; k++) {
            key = order[k]; split(key, f, SUBSEP)
            m[key] = (n[key] > 0) ? sum[key] / n[key] : -1
            if (f[3] == "gpu1" && f[4] == "gpu") base[f[1], f[2]] = m[key]
        }
        for (k = 1; k <= nk; k++) {
            key = order[k]; split(key, f, SUBSEP)
            c = n[key] + 0
            if (c == 0) { printf "%s,%s,%s,%s,0,%s,,,,,,,,\n", f[1], f[2], f[3], f[4], asked[key]; continue }
            mean = m[key]
            var = (c > 1) ? (sq[key] - c * mean * mean) / (c - 1) : 0
            if (var < 0) var = 0
            sd = sqrt(var)
            ci = (c > 1) ? tq(c - 1) * sd / sqrt(c) : 0
            per = (iters[key] > 0) ? 1000 * mean / iters[key] : 0
            b = base[f[1], f[2]]
            sp = (b > 0) ? sprintf("%.3f", b / mean) : ""
            printf "%s,%s,%s,%s,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.4f,%s\n", f[1], f[2], f[3], f[4], c, asked[key], mean, sd, ci, mean - ci, mean + ci, 100 * sd / mean, per, sp
        }
    }' > "$OUT.tmp"

# Sorted: benchmark, precision, scenario (gpu1 n1 n2 n3), config order of the jobs.
{
    head -n 1 "$OUT.tmp"
    tail -n +2 "$OUT.tmp" | awk -F, '
        BEGIN { split("gpu off static_threshold static_profiled dynamic_threshold dynamic_profiled", c, " ")
                for (i in c) rank[c[i]] = i }
        { print $1 "," $2 "," $3 "," (($4 in rank) ? rank[$4] : 9) "\t" $0 }' |
        sort -t, -k1,1 -k2,2 -k3,3 -k4,4n | cut -f2-
} > "$OUT"
rm -f "$OUT.tmp"

column -t -s, "$OUT" 2>/dev/null || cat "$OUT"
echo ""
echo "Written: $OUT  (ci95 = half-width of the 95 % confidence interval of the mean, Student's t)"
