from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_tools(session):
    @tool
    def build_dtm_tool(ngram_range: List[int] = [1, 1], max_features: Optional[int] = None) -> dict:
        """Builds the document-term matrix (DTM) from session.dataframe text.
        
        Fits a CountVectorizer on the documents, stores the resulting sparse feature 
        matrix in session.feature_matrix, feature names in session.feature_names, 
        labels in session.labels/categories, and the fitted vectorizer in 
        session.artifacts["count_vectorizer"]. Reports matrix sparsity.

        Args:
            ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted, e.g. [1, 1].
            max_features: Optional limit on vocabulary size to keep only top features by frequency.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}
        
        df = session.dataframe
        texts = df["text"].fillna("").tolist()
        
        ngram_tuple = (ngram_range[0], ngram_range[1])
        
        vectorizer = CountVectorizer(ngram_range=ngram_tuple, max_features=max_features)
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()
        
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        
        if "category_name" in df.columns:
            labels = df["category_name"].values
            session.set_labels(labels)
        
        session.artifacts["count_vectorizer"] = vectorizer
        
        non_zero = feature_matrix.nnz
        n_rows, n_cols = feature_matrix.shape
        total_elements = n_rows * n_cols
        sparsity_pct = float(100 * (1 - non_zero / total_elements)) if total_elements > 0 else 0.0
        
        summary = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct
        }
        
        result_id = session.next_result_id("dtm")
        summary["result_id"] = result_id
        
        session.store_result("build_dtm_tool", {"ngram_range": ngram_range, "max_features": max_features}, summary)
        
        return summary

    @tool
    def term_frequency_tool() -> dict:
        """Aggregates term frequencies (column sums) across all documents in session.feature_matrix."""
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built yet. Run build_dtm_tool first."}
        
        counts = session.feature_matrix.sum(axis=0).A1
        term_freqs = {name: int(count) for name, count in zip(session.feature_names, counts)}
        
        sorted_freqs = dict(sorted(term_freqs.items(), key=lambda x: x[1], reverse=True))
        
        summary = {
            "term_frequencies": sorted_freqs
        }
        
        result_id = session.next_result_id("term_freq")
        summary["result_id"] = result_id
        
        session.store_result("term_frequency_tool", {}, summary)
        
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Renders a heatmap of a positional slice of the global document-term matrix.

        Args:
            n_terms: Number of terms (columns) to include.
            n_documents: Number of documents (rows) to include.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built yet. Run build_dtm_tool first."}
        
        if not isinstance(n_terms, int) or n_terms <= 0:
            return {"error": "Parameter 'n_terms' must be a positive integer."}
        if not isinstance(n_documents, int) or n_documents <= 0:
            return {"error": "Parameter 'n_documents' must be a positive integer."}
        
        mat = session.feature_matrix
        names = session.feature_names
        
        # Slice matrix and terms
        doc_slice = min(n_documents, mat.shape[0])
        term_slice = min(n_terms, mat.shape[1])
        
        sliced_matrix = mat[0:doc_slice, 0:term_slice].toarray()
        sliced_names = names[0:term_slice]
        
        # Generate heatmap figure
        fig, ax = plt.subplots(figsize=(max(6, term_slice * 0.8), max(4, doc_slice * 0.5)))
        sns.heatmap(sliced_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=sliced_names, yticklabels=[str(i) for i in range(doc_slice)], ax=ax)
        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()
        
        session.pending_figure = fig
        
        summary = {
            "n_terms": term_slice,
            "n_documents": doc_slice,
            "matrix_slice": sliced_matrix.tolist(),
            "terms": sliced_names
        }
        
        result_id = session.next_result_id("dtm_heatmap")
        summary["result_id"] = result_id
        
        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
