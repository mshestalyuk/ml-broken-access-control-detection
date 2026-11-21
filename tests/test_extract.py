from pathlib import Path

from bacdetect.extract import changed_functions, functions_in_source, parse_diff

DIFF = """\
diff --git a/app/views.py b/app/views.py
index 1111111..2222222 100644
--- a/app/views.py
+++ b/app/views.py
@@ -6,0 +7,2 @@ def unchanged():
+    invoice = Invoice.query.get_or_404(invoice_id)
+    return invoice.to_json()
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old
+new
diff --git a/gone.py b/gone.py
--- a/gone.py
+++ /dev/null
@@ -1,2 +0,0 @@
-x = 1
-y = 2
"""

SOURCE = """\
def unchanged():
    return 1


@app.get("/invoices/<int:invoice_id>")
@login_required
def get_invoice(invoice_id):
    invoice = Invoice.query.get_or_404(invoice_id)
    return invoice.to_json()
"""


def test_parse_diff():
    changed = parse_diff(DIFF)
    assert changed == {"app/views.py": {7, 8}, "README.md": {1}}


def test_functions_include_decorators():
    fns = {f.name: f for f in functions_in_source(SOURCE)}
    assert fns["get_invoice"].start == 5
    assert fns["get_invoice"].code.startswith('@app.get("/invoices')


def test_changed_functions(tmp_path: Path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "views.py").write_text(SOURCE)
    (tmp_path / "README.md").write_text("new\n")
    fns = changed_functions(DIFF, tmp_path)
    assert [f.name for f in fns] == ["get_invoice"]


def test_syntax_error_is_skipped():
    assert functions_in_source("def broken(:\n") == []


def test_git_diff_rejects_option_like_refs():
    import pytest

    from bacdetect.extract import git_diff

    with pytest.raises(ValueError):
        git_diff("--output=/tmp/x", "HEAD")
