"""Build scene summaries and fold-level results from the complete prediction audit."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    scenes = {"IndianPines": ("Indian Pines", 4), "PaviaU": ("Pavia University", 5), "Salinas": ("Salinas", 5)}
    variants = ("clean", "pids")
    if (audit["status"], audit["runs"], audit["prediction_files"], audit["matched_pairs"]) != ("PASS", 28, 448, 14) or audit["incomplete_pairs"]:
        parser.exit(1, "Full prediction audit required for the manuscript table.\n")
    records = {(r["dataset"], r["fold"], r["variant"]): r for r in audit["records"]}
    expected = {(s, f, v) for s, (_, n) in scenes.items() for f in range(n) for v in variants}
    assert set(records) == expected and len(audit["records"]) == len(expected)
    columns = (("Clean", "oa"), ("L5", "oa"), ("L5", "macro_f1"))
    tex = [r"\begin{table}[!htbp]", r"\centering",
           r"\caption{\label{tab:published-partitions}Classification performance on published spatial partitions.}",
           r"\JARStableformat", r"\begin{tabularx}{\textwidth}{lYYYY}", r"\toprule"]
    prefix = r"\ifdefined\JARSClean\else\rowcolor{yellow}\fi "
    endrow = chr(92) * 2
    tex.append(prefix + r"Scene & Training & Clean OA (\%) & L5 OA (\%) & L5 macro-F1 (\%) " + endrow)
    flat = []
    for scene, (label, folds) in scenes.items():
        tex.append(r"\midrule")
        for variant in variants:
            group = [records[scene, f, variant] for f in range(folds)]
            assert [r["seed"] for r in group] == list(range(23, 23 + folds))
            cells = []
            for condition, metric in columns:
                values = [r["results"][condition][metric] for r in group]
                mean, sd = statistics.mean(values), statistics.stdev(values)
                summary = audit["aggregate"][scene][variant][condition][metric]
                assert all(math.isclose(v, summary[k], abs_tol=1e-12) for v,k in [(mean,"mean"),(sd,"sd")])
                cells.append(f"${100*mean:.2f}\\pm{100*sd:.2f}$")
            tex.append(prefix + " & ".join([label, "Clean" if variant == "clean" else "PIDS", *cells]) + " " + endrow)
            for r in group:
                for condition, metrics in r["results"].items():
                    flat.append(dict(scene=label, fold=r["fold"]+1, seed=r["seed"], training=variant, condition=condition, **metrics))
    tex.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"table11_published.tex").write_bytes(("\n".join(tex)+"\n").encode())
    with (args.output/"published_folds.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


if __name__ == "__main__":
    main()
