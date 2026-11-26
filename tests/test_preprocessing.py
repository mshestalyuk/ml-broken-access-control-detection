from bacdetect.preprocessing import clean_code, split_identifier, strip_comments, tokenize


def test_strips_python_and_c_comments_but_keeps_strings():
    code = 'x = "# not a comment"  # real comment\n/* block */ y = 1; // line\n'
    out = strip_comments(code)
    assert "# not a comment" in out
    assert "real comment" not in out
    assert "block" not in out and "line" not in out


def test_docstrings_removed_line_count_preserved():
    code = 'def f():\n    """doc\n    string"""\n    return 1\n'
    out = strip_comments(code)
    assert "doc" not in out
    assert out.count("\n") == code.count("\n")


def test_c_preprocessor_directive_kept():
    assert "#include" in strip_comments("#include <stdio.h>\n")


def test_clean_code_normalizes_whitespace():
    assert clean_code("a = 1   \n\n\n\tb = 2\n") == "a = 1\n    b = 2"


def test_split_identifier():
    assert split_identifier("owner_id") == ["owner", "id"]
    assert split_identifier("findById") == ["find", "by", "id"]


def test_tokenize_keeps_identifiers_and_subtokens():
    toks = tokenize("x = Invoice.query.filter_by(owner_id=current_user.id)")
    assert "owner_id" in toks and "owner" in toks
    assert "current_user" in toks
    assert "==" not in toks and "=" in toks
    assert tokenize("n = 42") == ["n", "=", "<num>"]
