"""Score changed functions and build / post the PR comment (soft gate, never blocks merge)."""

import json
import os
import urllib.request
from dataclasses import dataclass

from bacdetect.config import COMMENT_MARKER
from bacdetect.extract import ChangedFunction
from bacdetect.features import extract_features


@dataclass
class Finding:
    fn: ChangedFunction
    risk: float
    flagged: bool
    reasons: list[str]


REASONS = {
    "id_lookup_without_owner_filter": "object fetched by id without an owner filter",
    "mutating_without_role_check": "PUT/PATCH/DELETE without role or ownership check",
}


def explain(code: str) -> list[str]:
    feats = extract_features(code)
    reasons = [text for key, text in REASONS.items() if feats[key]]
    authenticated = feats["has_auth_decorator"] or feats["uses_current_user"]
    if feats["uses_request_param"] and not authenticated:
        reasons.append("request parameter used, no authentication context")
    return reasons


def score(functions: list[ChangedFunction], artifact: dict) -> list[Finding]:
    if not functions:
        return []
    est, threshold = artifact["estimator"], artifact["meta"]["threshold"]
    proba = est.predict_proba([f.code for f in functions])[:, 1]
    return [
        Finding(fn=f, risk=float(p), flagged=p >= threshold, reasons=explain(f.code))
        for f, p in zip(functions, proba, strict=True)
    ]


def render_markdown(findings: list[Finding], meta: dict, seconds: float) -> str:
    flagged = sorted((f for f in findings if f.flagged), key=lambda f: -f.risk)
    head = [
        COMMENT_MARKER,
        "### bacdetect: Broken Access Control / IDOR check",
        "",
        f"Scanned **{len(findings)}** changed function(s), **{len(flagged)}** flagged "
        f"(model `{meta['model_name']}`, threshold {meta['threshold']:.2f}, {seconds:.1f}s).",
        "",
    ]
    if not flagged:
        return "\n".join(head + ["No risky access-control changes detected."])
    rows = ["| file | function | risk | why |", "|---|---|---|---|"]
    for f in flagged:
        why = "; ".join(f.reasons) or "model score"
        rows.append(f"| `{f.fn.path}:{f.fn.start}` | `{f.fn.name}` | {f.risk:.0%} | {why} |")
    tail = [
        "",
        "_Advisory only: this check does not block merge. Please review ownership "
        "and role checks in the functions above._",
    ]
    return "\n".join(head + rows + tail)


def _github(method: str, url: str, token: str, body: dict | None = None) -> dict | list:
    req = urllib.request.Request(
        url,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read() or b"{}")


def post_comment(markdown: str, repo: str, pr: int, token: str | None = None) -> str:
    """Create or update a single sticky comment identified by COMMENT_MARKER."""
    token = token or os.environ["GITHUB_TOKEN"]
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    comments = _github("GET", f"{api}/repos/{repo}/issues/{pr}/comments?per_page=100", token)
    existing = next((c for c in comments if COMMENT_MARKER in c.get("body", "")), None)
    if existing:
        _github("PATCH", existing["url"], token, {"body": markdown})
        return existing["html_url"]
    created = _github("POST", f"{api}/repos/{repo}/issues/{pr}/comments", token, {"body": markdown})
    return created["html_url"]
