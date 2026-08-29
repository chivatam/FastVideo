from research.br_vsa.assemble_results import FINAL_DECISION


def test_final_decision_is_protocol_phrase() -> None:
    assert FINAL_DECISION == (
        "DECISION: STOP — STATIC GLOBAL-BUDGET REDISTRIBUTION "
        "DOES NOT RECOVER ENOUGH QUALITY"
    )
