from pathlib import Path

import pytest

from bacdetect.config import SEED_DATASET
from bacdetect.data.bac_idor import load_bac_idor

VULN = """
@app.get("/invoices/<int:invoice_id>")
@login_required
def get_invoice(invoice_id):
    # TODO: check owner
    return Invoice.query.get_or_404(invoice_id).to_json()
"""

FIXED = """
@app.get("/invoices/<int:invoice_id>")
@login_required
def get_invoice(invoice_id):
    invoice = Invoice.query.filter_by(id=invoice_id, owner_id=current_user.id).first_or_404()
    return invoice.to_json()
"""


@pytest.fixture
def seed_dataset():
    return load_bac_idor(Path(SEED_DATASET), n_folds=3)
