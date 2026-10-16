from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_index_a: int, doc_index_b: int) -> dict:
        """Computes the cosine similarity between two documents in the global DTM.

        Args:
            doc_index_a: Index of the first document.
            doc_index_b: Index of the second document.
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix built yet. Run build_dtm_tool first."}
        
        if doc_index_a < 0 or doc_index_a >= session.feature_matrix.shape[0] or \
           doc_index_b < 0 or doc_index_b >= session.feature_matrix.shape[0]:
            return {"error": f"Document index out of range. Matrix has {session.feature_matrix.shape[0]} documents."}
        
        # Extract the two rows
        vec_a = session.feature_matrix[doc_index_a]
        vec_b = session.feature_matrix[doc_index_b]
        
        # Compute cosine similarity
        similarity = cosine_similarity(vec_a, vec_b)[0][0]
        
        summary = {
            "doc_index_a": doc_index_a,
            "doc_index_b": doc_index_b,
            "similarity": float(similarity)
        }
        
        result_id = session.next_result_id("cosine_sim")
        summary["result_id"] = result_id
        
        session.store_result(
            "cosine_similarity_tool",
            {"doc_index_a": doc_index_a, "doc_index_b": doc_index_b},
            summary
        )
        
        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Computes a feature-vs-feature correlation heatmap over the top-20 terms by variance.

        Computes the Pearson correlation matrix across the top-20 terms by variance
        and sets a heatmap of this matrix on session.pending_figure.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built yet. Run build_dtm_tool first."}
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # 1. Compute variance for each term
        dense_matrix = session.feature_matrix.toarray()
        variances = np.var(dense_matrix, axis=0)
        
        # 2. Get top 20 terms by variance
        top_indices = np.argsort(variances)[-20:][::-1]
        top_terms = [session.feature_names[i] for i in top_indices]
        
        # 3. Compute Pearson correlation matrix
        subset_matrix = dense_matrix[:, top_indices]
        corr_matrix = np.corrcoef(subset_matrix, rowvar=False)
        
        # Handle NaN values (e.g., from zero-variance terms)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # 4. Build heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, xticklabels=top_terms, yticklabels=top_terms, ax=ax)
        session.pending_figure = fig
        
        # 5. Prepare summary
        summary = {
            "terms": top_terms,
            "correlation_matrix": corr_matrix.tolist()
        }
        
        result_id = session.next_result_id("feat_corr")
        summary["result_id"] = result_id
        
        session.store_result(
            "feature_correlation_matrix_tool",
            {},
            summary
        )
        
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
