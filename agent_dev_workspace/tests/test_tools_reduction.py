import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools

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
    reduction_tools = make_reduction_tools(session)
    
    # Setup: Load data and build DTM
    load_dataset_tool.invoke({"file_path": "agent_dev/sample_fixture.csv", "text_column": "text", "label_column": "label"})
    
    # Build DTM
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return {t.name: t for t in dtm_tools + reduction_tools}

def test_reduce_dimensions_tool(session, tools):
    reduce_dim = tools["reduce_dimensions_tool"]
    
    # Known Answer 7: PCA
    res_pca = reduce_dim.invoke({"method": "pca"})
    coords_pca = np.array(res_pca["coordinates"])
    assert coords_pca.shape == (8, 2)
    assert res_pca["explained_variance"] == pytest.approx([0.7532, 0.1965], abs=1e-3)
    
    # Known Answer 8: t-SNE (perplexity=2, random_state=42)
    res_tsne = reduce_dim.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    coords_tsne = np.array(res_tsne["coordinates"])
    assert coords_tsne.shape == (8, 2)
    
    # Known Answer 9: UMAP (n_neighbors=3, random_state=42)
    res_umap = reduce_dim.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    coords_umap = np.array(res_umap["coordinates"])
    assert coords_umap.shape == (8, 2)
    
    # Verify plotting requirement
    assert session.pending_figure is not None

def test_binarize_labels_tool(session, tools):
    binarize_labels = tools["binarize_labels_tool"]
    
    # Run the tool
    result = binarize_labels.invoke({})
    
    # Verify summary
    assert result["n_classes"] == 2
    assert result["shape"] == (8, 2)
    
    # Verify artifacts
    binary_matrix = session.artifacts["binarized_labels"]
    assert binary_matrix.shape == (8, 2)
    
    # Verify encoding (catA=0, catB=1 in fixture, so label_catA and label_catB columns)
    # Row 0 is catA -> [1, 0]
    # Row 3 is catB -> [0, 1]
    assert np.array_equal(binary_matrix[0], [1, 0])
    assert np.array_equal(binary_matrix[3], [0, 1])
