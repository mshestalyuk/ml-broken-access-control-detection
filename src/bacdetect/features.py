import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from bacdetect.preprocessing import clean_code, tokenize


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


AUTH_DECORATOR = _rx(
    r"@\s*(login_required|permission_required|requires?_auth|jwt_required|auth_required)",
    r"@\s*(PreAuthorize|Secured|RolesAllowed|Authorize)\b",
    r"Depends\(\s*get_current_(active_)?user",
    r"\b(isAuthenticated|ensureAuthenticated|requireAuth)\b",
)
CURRENT_USER = _rx(
    r"\bcurrent_user\b",
    r"\brequest\.user\b",
    r"\breq\.user\b",
    r"\bg\.user\b",
    r"\bsession\s*\[",
    r"\bget_current_user\b",
    r"\bgetPrincipal\b",
    r"\bauth\(\)",
)
REQUEST_PARAM = _rx(
    r"\brequest\.(args|form|json|GET|POST|params|query|body)\b",
    r"\breq\.(params|query|body)\b",
    r"@(PathVariable|RequestParam)\b",
    r"<\s*(int|uuid|string)?:?\w*id\s*>",
    r"\{\w*id\}",
)
DB_LOOKUP_BY_ID = _rx(
    r"\.get_or_404\(",
    r"\.query\.get\(",
    r"\.objects\.get\(\s*(pk|id)\s*=",
    r"get_object_or_404\(",
    r"\bfind_?by_?id\b",
    r"\.findOne\(",
    r"\.findByPk\(",
    r"filter_by\(\s*id\s*=",
    r"\bsession\.get\(",
    r"WHERE\s+id\s*=",
)
OWNER_FILTER = _rx(
    r"\b(owner|user|author|created_by|tenant|account)(_id)?\s*=\s*(current_user|request\.user|"
    r"req\.user|g\.user|user)",
    r"\.(owner|user|author)(_id)?\s*(==|!=)\s*(current_user|request\.user|req\.user|g\.user)",
    r"\b(check_|has_)?(object_)?permission\(",
    r"\bauthorize\(",
    r"\bcan_(view|edit|access)\b",
)
ROLE_CHECK = _rx(
    r"\b(has_role|is_admin|is_staff|is_superuser|require_role|roles?_required)\b",
    r"\bhasRole\b",
    r"\bcheck_permission\b",
    r"\bPermissionDenied\b",
    r"\babort\(\s*403",
    r"\b403\b",
    r"\bForbidden\b",
)
MUTATING = _rx(
    r"\.(put|patch|delete)\(",
    r"methods\s*=\s*\[[^\]]*(PUT|PATCH|DELETE)",
    r"@(Put|Patch|Delete)Mapping",
    r"\.(delete|update|destroy|save)\(",
)
BRANCH = re.compile(r"\b(if|elif|for|while|case|catch|except|and|or)\b|&&|\|\||\?")
FUNC_SIG = re.compile(r"(?:def|function)\s+\w+\s*\(([^)]*)\)|\w+\s*\(([^)]*)\)\s*\{")

FEATURE_NAMES = [
    "has_auth_decorator",
    "uses_current_user",
    "uses_request_param",
    "db_lookup_by_id",
    "has_owner_filter",
    "has_role_check",
    "is_mutating",
    "id_lookup_without_owner_filter",
    "mutating_without_role_check",
    "n_lines",
    "n_params",
    "n_tokens",
    "unique_token_ratio",
    "cyclomatic_complexity",
]


def _count_params(code: str) -> int:
    m = FUNC_SIG.search(code)
    if not m:
        return 0
    args = (m.group(1) or m.group(2) or "").strip()
    return 0 if not args else len([a for a in args.split(",") if a.strip()])


def extract_features(code: str) -> dict[str, float]:
    clean = clean_code(code)
    tokens = tokenize(clean, subtokens=False)

    auth = bool(AUTH_DECORATOR.search(clean))
    user = bool(CURRENT_USER.search(clean))
    lookup = bool(DB_LOOKUP_BY_ID.search(clean))
    owner = bool(OWNER_FILTER.search(clean))
    role = bool(ROLE_CHECK.search(clean))
    mutating = bool(MUTATING.search(clean))

    return {
        "has_auth_decorator": float(auth),
        "uses_current_user": float(user),
        "uses_request_param": float(bool(REQUEST_PARAM.search(clean))),
        "db_lookup_by_id": float(lookup),
        "has_owner_filter": float(owner),
        "has_role_check": float(role),
        "is_mutating": float(mutating),
        # The core IDOR signal: object fetched by id, no ownership check.
        "id_lookup_without_owner_filter": float(lookup and not owner),
        "mutating_without_role_check": float(mutating and not (role or owner)),
        "n_lines": float(clean.count("\n") + 1 if clean else 0),
        "n_params": float(_count_params(clean)),
        "n_tokens": float(len(tokens)),
        "unique_token_ratio": len(set(tokens)) / len(tokens) if tokens else 0.0,
        "cyclomatic_complexity": float(1 + len(BRANCH.findall(clean))),
    }


class BacIdorFeatures(BaseEstimator, TransformerMixin):
    """Turns raw code strings into a dense matrix of FEATURE_NAMES columns."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn convention
        return self

    def transform(self, X):  # noqa: N803
        rows = [extract_features(code) for code in X]
        return np.array([[r[name] for name in FEATURE_NAMES] for r in rows], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES, dtype=object)
