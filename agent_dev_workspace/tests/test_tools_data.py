from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_list_files_tool():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")
    
    result = list_files_tool.invoke({"subdirectory": "newdataset"})
    
    # Check Known Answer 11
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]

def test_inspect_data_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")
    result = inspect_data_tool.invoke({"n": 3})
    
    assert result["total_rows"] == 8
    assert result["columns"] == ["text", "category_name", "category"]
    assert result["n_inspected"] == 3
    assert len(result["rows"]) == 3

def test_check_missing_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")
    result = check_missing_tool.invoke({})
    
    # Check Known Answer 2: Exactly 1 row has missing/empty text
    assert result["n_missing"] == 1
    assert result["total_rows"] == 8

def test_check_duplicates_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")
    
    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check["n_duplicates"] == 2
    
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["total_rows"] == 6
    assert res_drop["n_dropped"] == 2

def test_sample_data_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")
    
    # Check Known Answer 13: n=4, random_state=42 returns rows [1, 5, 0, 7]
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert result["n_sampled"] == 4

def test_tokenize_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    result = tokenize_tool.invoke({})
    
    # Check Known Answer 10 token lists
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
