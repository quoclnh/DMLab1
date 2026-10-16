import pytest
import numpy as np
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
    
    test_cases = [
        (0, 1, 0.9847),
        (0, 3, 0.0909),
        (3, 4, 0.9847),
        (0, 7, 0.0)
    ]
    
    for idx_a, idx_b, expected_sim in test_cases:
        result = cosine_sim.invoke({"doc_index_a": idx_a, "doc_index_b": idx_b})
        assert result["similarity"] == pytest.approx(expected_sim, abs=1e-3)

def test_feature_correlation_matrix_tool(session, tools):
    feat_corr = tools["feature_correlation_matrix_tool"]
    
    # Known Answer 19: Feature correlation matrix
    result = feat_corr.invoke({})
    
    terms = result["terms"]
    corr_matrix = np.array(result["correlation_matrix"])
    
    # Check terms order (descending variance: alpha, gamma, delta, beta, always)
    assert terms == ["alpha", "gamma", "delta", "beta", "always"]
    
    # Check specific correlations
    # alpha/beta (0.8885)
    alpha_idx = terms.index("alpha")
    beta_idx = terms.index("beta")
    assert corr_matrix[alpha_idx, beta_idx] == pytest.approx(0.8885, abs=1e-3)
    
    # gamma/delta (0.8307)
    gamma_idx = terms.index("gamma")
    delta_idx = terms.index("delta")
    assert corr_matrix[gamma_idx, delta_idx] == pytest.approx(0.8307, abs=1e-3)
