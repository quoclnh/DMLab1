import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def _setup_session():
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade if t.name == "load_dataset_tool")
    
    # Load sample fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool

def test_mine_patterns_zero_terms_survived():
    """Known answer #17: variance and term_frequency keep 0 terms on this fixture."""
    session, mine_patterns_tool = _setup_session()
    
    # Check variance method on catA
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var.get("patterns") == []
    assert res_var.get("note") == "no terms survived filtering"

    # Check term_frequency method on catB
    res_tf = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tf.get("patterns") == []
    assert res_tf.get("note") == "no terms survived filtering"

def test_mine_patterns_fpgrowth_tfidf():
    """Known answer #17: tfidf filtering + fpgrowth on catA and catB."""
    session, mine_patterns_tool = _setup_session()

    # catA -> keeps alpha -> support 3
    res_cata = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    patterns_a = res_cata.get("patterns", [])
    assert len(patterns_a) == 1
    assert patterns_a[0]["pattern"] == ["alpha"]
    assert pytest.approx(patterns_a[0]["support"]) == 3.0

    # catB -> keeps gamma -> support 4
    res_catb = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    patterns_b = res_catb.get("patterns", [])
    assert len(patterns_b) == 1
    assert patterns_b[0]["pattern"] == ["gamma"]
    assert pytest.approx(patterns_b[0]["support"]) == 4.0

def test_mine_patterns_topk_and_maxfpgrowth():
    """Known answer #20: topk and maxfpgrowth behavior."""
    session, mine_patterns_tool = _setup_session()

    # algorithm="topk" (k=1)
    res_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    patterns_topk = res_topk.get("patterns", [])
    assert len(patterns_topk) == 1
    assert patterns_topk[0]["pattern"] == ["alpha"]
    assert pytest.approx(patterns_topk[0]["support"]) == 3.0

    # algorithm="maxfpgrowth" returns 0 patterns for both catA and catB at min_sup=1 and min_sup=4
    for cat in ["catA", "catB"]:
        for min_sup_val in [1, 4]:
            res_max = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": min_sup_val
            })
            assert res_max.get("patterns") == []
