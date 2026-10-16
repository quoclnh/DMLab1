from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal, Optional

def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"],
        random_state: Optional[int] = None,
        perplexity: int = 30,
        n_neighbors: int = 15
    ) -> dict:
        """Reduces the dimensionality of the global DTM to 2D and plots the result.

        Args:
            method: The dimensionality reduction method to use ("pca", "tsne", or "umap").
            random_state: Seed for reproducible output.
            perplexity: Perplexity parameter for t-SNE.
            n_neighbors: Number of neighbors for UMAP.
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix built yet. Run build_dtm_tool first."}
        
        dense_matrix = session.feature_matrix.toarray()
        n_docs = dense_matrix.shape[0]
        
        if method == "pca":
            model = PCA(n_components=2, random_state=random_state)
            coords = model.fit_transform(dense_matrix)
            explained_variance = model.explained_variance_ratio_.tolist()
        elif method == "tsne":
            if perplexity <= 0 or perplexity >= n_docs:
                return {"error": f"Perplexity must be a positive integer less than the number of documents ({n_docs})."}
            model = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
            coords = model.fit_transform(dense_matrix)
            explained_variance = None
        elif method == "umap":
            if n_neighbors <= 0 or n_neighbors > n_docs:
                return {"error": f"n_neighbors must be a positive integer at most the number of documents ({n_docs})."}
            try:
                import umap
            except ImportError:
                return {"error": "UMAP is not installed. Please install 'umap-learn' to use this method."}
            model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
            coords = model.fit_transform(dense_matrix)
            explained_variance = None
        else:
            return {"error": f"Unknown method: {method}"}
        
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = session.dataframe["category_name"] if "category_name" in session.dataframe else session.labels
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, ax=ax)
        ax.set_title(f"{method.upper()} Reduction")
        session.pending_figure = fig
        
        summary = {
            "method": method,
            "coordinates": coords.tolist(),
            "explained_variance": explained_variance
        }
        
        result_id = session.next_result_id("dim_reduction")
        summary["result_id"] = result_id
        
        session.store_result(
            "reduce_dimensions_tool",
            {"method": method, "random_state": random_state, "perplexity": perplexity, "n_neighbors": n_neighbors},
            summary
        )
        
        return summary

    return [reduce_dimensions_tool]
