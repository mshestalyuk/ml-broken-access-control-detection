"""Code cleaning and tokenization (language-agnostic: works for Python, C, JS, Java).

Steps:
  1. strip comments (string literals are kept intact, so '#' inside a string survives)
  2. standardize whitespace
  3. tokenize into keywords / identifiers / operators, keeping identifier names
     (owner_id, current_user carry authorization signal) plus their sub-tokens.
"""
import re

# One pass: match strings first so comment markers inside them are ignored.
_STRING_OR_COMMENT = re.compile(
    r"""
    (?P<string>
        \"\"\"[\s\S]*?\"\"\"          # python triple double
      | '''[\s\S]*?'''                # python triple single
      | "(?:\\.|[^"\\\n])*"           # double-quoted
      | '(?:\\.|[^'\\\n])*'           # single-quoted
      | `(?:\\.|[^`\\])*`             # JS template literal
    )
    | (?P<comment>
        /\*[\s\S]*?\*/                # C block comment
      | //[^\n]*                      # C / JS / Java line comment
      | \#(?!include|define|if|endif|else|elif|pragma|undef)[^\n]*  # python comment
    )
    """,
    re.VERBOSE,
)

_TOKEN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"  # identifiers / keywords
    r"|\d+(?:\.\d+)?"  # numbers
    r"|==|!=|<=|>=|->|=>|&&|\|\||::|\*\*"
    r"|[^\sA-Za-z0-9_]"  # single-char punctuation
)
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def strip_comments(code: str) -> str:
    def _repl(m: re.Match[str]) -> str:
        if m.group("comment") is not None:
            # keep newlines so line numbers stay stable
            return "\n" * m.group("comment").count("\n")
        s = m.group("string")
        # Docstrings are documentation, not logic: drop triple-quoted strings
        if s.startswith(('"""', "'''")):
            return "\n" * s.count("\n")
        return s

    return _STRING_OR_COMMENT.sub(_repl, code)


def normalize_whitespace(code: str) -> str:
    lines = (line.rstrip() for line in code.replace("\t", "    ").splitlines())
    return "\n".join(line for line in lines if line.strip())


def clean_code(code: str) -> str:
    return normalize_whitespace(strip_comments(code))


def split_identifier(token: str) -> list[str]:
    """owner_id -> [owner, id]; findById -> [find, by, id]."""
    parts: list[str] = []
    for chunk in token.split("_"):
        parts.extend(p.lower() for p in _CAMEL.findall(chunk))
    return [p for p in parts if p]


def tokenize(code: str, subtokens: bool = True) -> list[str]:
    """Tokenize already-cleaned code. Identifiers are lowercased; sub-tokens appended."""
    out: list[str] = []
    for tok in _TOKEN.findall(code):
        if tok[0].isalpha() or tok[0] == "_":
            low = tok.lower()
            out.append(low)
            if subtokens:
                parts = split_identifier(tok)
                if len(parts) > 1:
                    out.extend(parts)
        elif tok[0].isdigit():
            out.append("<num>")
        else:
            out.append(tok)
    return out


def code_tokenizer(code: str) -> list[str]:
    """Entry point used by TfidfVectorizer (must be a picklable top-level function)."""
    return tokenize(clean_code(code))
