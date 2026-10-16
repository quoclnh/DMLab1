import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

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

def test_build_dtm_tool():
    session = SessionState()
    load_sample_fixture(session)
    
    build_dtm_tool = get_tool(session, "build_dtm_tool")
    result = build_dtm_tool.invoke({})
    
    # Check Known Answer 4: Vocabulary and sparsity
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert result["n_rows"] == 8
    assert result["n_cols"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

def test_term_frequency_tool():
    session = SessionState()
    load_sample_fixture(session)
    
    build_dtm_tool = get_tool(session, "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    term_frequency_tool = get_tool(session, "term_frequency_tool")
    result = term_frequency_tool.invoke({})
    
    # Check Known Answer 14: Total term frequencies
    expected_freqs = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3
    }
    
    assert result["term_frequencies"] == expected_freqs

def test_dtm_heatmap_tool():
    session = SessionState()
    load_sample_fixture(session)
    
    build_dtm_tool = get_tool(session, "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    dtm_heatmap_tool = get_tool(session, "dtm_heatmap_tool")
    
    # Check Known Answer 21: n_terms=2, n_documents=3 should return top-left 3x2 corner
    result = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    
    expected_slice = [
        [3, 1],
        [2, 1],
        [1, 1]
    ]
    
    assert result["matrix_slice"] == expected_slice
    assert result["terms"] == ["alpha", "always"]
    assert session.pending_figure is not None
