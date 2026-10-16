import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

@pytest.fixture
def session():
    return SessionState()

@pytest.fixture
def tools(session):
    # Load premade tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load student tools
    dtm_tools = make_dtm_tools(session)
    exploration_tools = make_exploration_tools(session)
    
    # Setup: Load data and build DTM
    load_dataset_tool.invoke({"file_path": "agent_dev/sample_fixture.csv", "text_column": "text", "label_column": "label"})
    
    # Build DTM
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return {t.name: t for t in dtm_tools + exploration_tools}

def test_cosine_similarity_tool(session, tools):
    cosine_sim = tools["cosine_similarity_tool"]
    
    # Known Answer 16: Cosine similarity pairs
    # 0 vs 1: 0.9847
    # 0 vs 3: 0.0909
    # 3 vs 4: 0.9847
    # 0 vs 7: 0.0
    
    test_cases = [
        (0, 1, 0.9847),
        (0, 3, 0.0909),
        (3, 4, 0.9847),
        (0, 7, 0.0)
    ]
    
    for idx_a, idx_b, expected_sim in test_cases:
        result = cosine_sim.invoke({"doc_index_a": idx_a, "doc_index_b": idx_b})
        assert result["similarity"] == pytest.approx(expected_sim, abs=1e-3)
