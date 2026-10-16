import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def load_sample_fixture(session):
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session

def get_tool(session, tool_name):
    tools = make_tools(session)
    return next(t for t in tools if t.name == tool_name)

def test_list_files_tool():
    session = SessionState()
    list_files_tool = get_tool(session, "list_files_tool")
    
    result = list_files_tool.invoke({"subdirectory": "newdataset"})
    
    # Check Known Answer 11
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]

def test_inspect_data_tool():
    session = SessionState()
    load_sample_fixture(session)
    
    inspect_data_tool = get_tool(session, "inspect_data_tool")
    result = inspect_data_tool.invoke({"n": 3})
    
    assert result["total_rows"] == 8
    assert result["columns"] == ["text", "category_name", "category"]
    assert result["n_inspected"] == 3
    assert len(result["rows"]) == 3

def test_check_missing_tool():
    session = SessionState()
    load_sample_fixture(session)
    
    check_missing_tool = get_tool(session, "check_missing_tool")
    result = check_missing_tool.invoke({})
    
    # Check Known Answer 2: Exactly 1 row has missing/empty text
    assert result["n_missing"] == 1
    assert result["total_rows"] == 8

def test_check_duplicates_tool():
    session = SessionState()
    load_sample_fixture(session)
    
    check_duplicates_tool = get_tool(session, "check_duplicates_tool")
    
    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check["n_duplicates"] == 2
    
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["total_rows"] == 6
    assert res_drop["n_dropped"] == 2

def test_sample_data_tool():
    session = SessionState()
    load_sample_fixture(session)
    
    sample_data_tool = get_tool(session, "sample_data_tool")
    
    # Check Known Answer 13: n=4, random_state=42 returns rows [1, 5, 0, 7]
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert result["n_sampled"] == 4

def test_tokenize_tool():
    session = SessionState()
    load_sample_fixture(session)
    
    tokenize_tool = get_tool(session, "tokenize_tool")
    result = tokenize_tool.invoke({})
    
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]
    
    assert session.dataframe["unigrams"].tolist() == expected_tokens
    assert result["n_documents"] == 8

def test_describe_data_tool():
    session = SessionState()
    load_sample_fixture(session)
    
    describe_data_tool = get_tool(session, "describe_data_tool")
    result = describe_data_tool.invoke({})
    
    # Check Known Answer 18
    expected_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
    assert session.dataframe["text_length"].tolist() == expected_lengths
    
    # Overall describe stats
    overall = result["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30
    
    # Per category stats
    cats = result["per_category_stats"]
    catA = cats["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["min"] == 0
    assert catA["max"] == 29
    
    catB = cats["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["min"] == 18
    assert catB["max"] == 30
    
    # Verify plotting requirement: session.pending_figure is set
    assert session.pending_figure is not None
