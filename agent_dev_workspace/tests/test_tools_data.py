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
    
    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Call inspect_data_tool with n=3
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")
    result = inspect_data_tool.invoke({"n": 3})
    
    assert result["total_rows"] == 8
    assert result["columns"] == ["text", "category_name", "category"]
    assert result["n_inspected"] == 3
    assert len(result["rows"]) == 3
