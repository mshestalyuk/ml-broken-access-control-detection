"""Command line: `bacdetect train | scan`."""

import argparse
import json
import sys
import time
from pathlib import Path

from bacdetect.config import MODELS_DIR, SEED_DATASET


def _load_dataset(args):
    if args.dataset == "codexglue":
        from bacdetect.data.codexglue import load_codexglue

        return load_codexglue(limit=args.limit)
    from bacdetect.data.bac_idor import load_bac_idor

    return load_bac_idor(Path(args.dataset_path))


def cmd_train(args) -> int:
    from bacdetect.evaluate import format_report, mcnemar
    from bacdetect.train import save_artifact, train_model

    ds = _load_dataset(args)
    print(f"dataset={ds.name} train={len(ds.train)} valid={len(ds.valid)} test={len(ds.test)}")

    results, report = [], []
    for name in args.models:
        r = train_model(name, ds, n_iter=args.n_iter, cv=args.cv, max_features=args.max_features)
        path = save_artifact(r, ds.name, Path(args.out))
        print(f"[{name}] test f1={r.test.f1:.3f} recall={r.test.recall:.3f} -> {path}")
        results.append(r)
        report.append(format_report(f"{name} (test)", r.test))

    if len(results) >= 2:
        a, b = sorted(results, key=lambda r: -r.test.f1)[:2]
        preds = [
            (r.estimator.predict_proba(ds.test.code)[:, 1] >= r.threshold).astype(int)
            for r in (a, b)
        ]
        stat = mcnemar(ds.test.label, *preds)
        report.append(f"## McNemar {a.model_name} vs {b.model_name}\n\n`{json.dumps(stat)}`")

    Path(args.out, "report.md").write_text("\n\n".join(report) + "\n")
    print(f"report -> {Path(args.out, 'report.md')}")
    return 0


def cmd_scan(args) -> int:
    from bacdetect.extract import changed_functions, git_diff
    from bacdetect.scan import post_comment, render_markdown, score
    from bacdetect.train import load_artifact

    start = time.perf_counter()
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"::warning::model {model_path} not found, skipping scan")
        return 0

    artifact = load_artifact(model_path)
    diff = (
        Path(args.diff_file).read_text()
        if args.diff_file
        else git_diff(args.base, args.head, Path(args.repo))
    )
    findings = score(changed_functions(diff, Path(args.repo)), artifact)
    md = render_markdown(findings, artifact["meta"], time.perf_counter() - start)

    if args.output:
        Path(args.output).write_text(md)
    if args.post:
        if not (args.github_repo and args.pr):
            print("--post needs --github-repo and --pr", file=sys.stderr)
            return 2
        print(f"comment: {post_comment(md, args.github_repo, args.pr)}")
    else:
        print("[dry-run] comment not posted, pass --post to publish:\n")
        print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bacdetect", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train and evaluate LR / RF / SVM")
    t.add_argument("--dataset", choices=["codexglue", "bac-idor"], default="bac-idor")
    t.add_argument("--dataset-path", default=str(SEED_DATASET), help="JSONL for bac-idor")
    t.add_argument("--models", nargs="+", default=["lr", "rf", "svm"], choices=["lr", "rf", "svm"])
    t.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations")
    t.add_argument("--cv", type=int, default=5)
    t.add_argument("--max-features", type=int, default=50_000)
    t.add_argument("--limit", type=int, default=None, help="rows per split (smoke runs)")
    t.add_argument("--out", default=str(MODELS_DIR))
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("scan", help="score functions changed in a PR")
    s.add_argument("--model", default=str(MODELS_DIR / "lr.joblib"))
    s.add_argument("--base", default="origin/main")
    s.add_argument("--head", default="HEAD")
    s.add_argument("--diff-file", help="read diff from file instead of running git")
    s.add_argument("--repo", default=".", help="checkout to read changed files from")
    s.add_argument("--output", help="write markdown report to this file")
    s.add_argument("--post", action="store_true", help="post comment (default: dry-run)")
    s.add_argument("--github-repo", help="owner/name")
    s.add_argument("--pr", type=int)
    s.set_defaults(func=cmd_scan)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
