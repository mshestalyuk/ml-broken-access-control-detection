from bacdetect.features import FEATURE_NAMES, BacIdorFeatures, extract_features
from tests.conftest import FIXED, VULN


def test_idor_signal_on_vulnerable_function():
    f = extract_features(VULN)
    assert f["has_auth_decorator"] == 1
    assert f["db_lookup_by_id"] == 1
    assert f["has_owner_filter"] == 0
    assert f["id_lookup_without_owner_filter"] == 1


def test_owner_filter_clears_idor_signal():
    f = extract_features(FIXED)
    assert f["has_owner_filter"] == 1
    assert f["uses_current_user"] == 1
    assert f["id_lookup_without_owner_filter"] == 0


def test_delete_without_role_check():
    code = (
        '@app.route("/u/<int:uid>", methods=["DELETE"])\n'
        "def rm(uid):\n"
        "    User.query.get(uid).delete()\n"
    )
    f = extract_features(code)
    assert f["is_mutating"] == 1
    assert f["mutating_without_role_check"] == 1
    check = "    if not current_user.is_admin:\n        abort(403)\n"
    guarded = code.replace("def rm(uid):\n", "def rm(uid):\n" + check)
    assert extract_features(guarded)["mutating_without_role_check"] == 0


def test_metrics():
    f = extract_features("def f(a, b):\n    if a and b:\n        return 1\n    return 0\n")
    assert f["n_params"] == 2
    assert f["n_lines"] == 4
    assert f["cyclomatic_complexity"] == 3  # 1 + if + and


def test_transformer_shape():
    X = BacIdorFeatures().fit_transform([VULN, FIXED, "int main(){return 0;}"])
    assert X.shape == (3, len(FEATURE_NAMES))
