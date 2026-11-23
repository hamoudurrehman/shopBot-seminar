"""Fast tests for our CUSTOM business rules. No API calls."""
from app.context import PATIENTS
from app.middleware import DOSE_QUESTION, pii_middleware
from app.rag import allowed_for, load_documents
from app.tools import is_urgent


def test_blood_pressure_threshold():
    assert is_urgent("p-001", "blood_pressure", "190/120")
    assert is_urgent("p-001", "blood_pressure", "150/112 mmHg")   # diastolic alone is enough
    assert not is_urgent("p-001", "blood_pressure", "128/82")


def test_glucose_thresholds():
    assert is_urgent("p-002", "glucose", "320")
    assert is_urgent("p-002", "glucose", "65 mg/dL")
    assert not is_urgent("p-002", "glucose", "120")


def test_dose_questions_are_caught():
    for q in ["Should I double my dose?", "Can I take 10mg instead?", "Can I stop taking amlodipine?"]:
        assert DOSE_QUESTION.search(q), q
    assert not DOSE_QUESTION.search("What's my target blood pressure?")


def test_patient_only_sees_own_notes():
    chunks = load_documents()
    visible = {c.metadata["source"] for c in chunks if allowed_for("p-001")(c)}
    assert "p-001.md" in visible and "blood_pressure.md" in visible
    assert "p-002.md" not in visible


def test_pii_regexes_match_our_formats():
    mrn, phone = pii_middleware()
    assert mrn.detector("my MRN is MRN-100001")
    assert phone.detector("call 555-0101")


def test_data_is_synthetic():
    assert all(p["name"].endswith("Demo") for p in PATIENTS.values())


def test_find_reading_in_patient_message():
    from app.middleware import find_reading
    assert find_reading("my bp is 190/120 what should I do?") == ("blood_pressure", "190/120")
    assert find_reading("my sugar is 320 this morning") == ("glucose", "320")
    assert find_reading("What's my blood pressure target?") is None
