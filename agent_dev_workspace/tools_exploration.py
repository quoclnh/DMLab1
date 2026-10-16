from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity

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

    return [cosine_similarity_tool]
