from agent_pipeline.session_state import SessionState
from tools_data import make_tools

def test_list_files_tool():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")
    
    result = list_files_tool.invoke({"subdirectory": "newdataset"})
    
    # Check Known Answer 11
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]
