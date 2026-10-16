import pytest
from agent_pipeline.workspace_loader import load_workspace_tools
from agent_pipeline.session_state import SessionState
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def load_sample_fixture(session):
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session

def get_tool(tools_list, tool_name):
    return next(t for t in tools_list if t.name == tool_name)

def test_variance_filter_tool():
    session = SessionState()
    load_sample_fixture(session)
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = get_tool(dtm_tools, "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = get_tool(filtering_tools, "variance_filter_tool")
    
    # Check Known Answer 5: threshold=0.15 keeps alpha, beta, delta, gamma and removes always
    result = variance_filter_tool.invoke({"threshold": 0.15})
    
    assert result["dropped_terms"] == ["always"]
    assert sorted(result["kept_terms"]) == sorted(["alpha", "beta", "delta", "gamma"])
    
    # Verify specific known variance values from Known Answer 5
    report_df = session.results_store[result["result_id"]]["full_report"]
    variances = dict(zip(report_df["term"], report_df["variance"]))
    
    assert variances["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert variances["always"] == pytest.approx(0.109375, abs=1e-3)
    assert variances["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert variances["delta"] == pytest.approx(0.25, abs=1e-3)
    assert variances["gamma"] == pytest.approx(1.109375, abs=1e-3)

def test_pearson_filter_tool():
    session = SessionState()
    load_sample_fixture(session)
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = get_tool(dtm_tools, "build_dtm_tool")
    build_dtm_tool.invoke({"ngram_range": [1, 1]})
    
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = get_tool(filtering_tools, "pearson_filter_tool")
    
    # Known Answer 6: target_class="catB", threshold=0.5 (absolute correlation < 0.5 dropped, e.g. "always" at 0.3780)
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})
    
    assert "always" in result["dropped_terms"]
    assert sorted(result["kept_terms"]) == sorted(["alpha", "beta", "delta", "gamma"])
    
    # Retrieve full_report to assert Pearson r for "delta"
    report_df = session.results_store[result["result_id"]]["full_report"]
    pearson_rs = dict(zip(report_df["term"], report_df["pearson_r"]))
    
    assert "delta" in pearson_rs
    assert pearson_rs["delta"] == pytest.approx(1.0000, abs=1e-3)
