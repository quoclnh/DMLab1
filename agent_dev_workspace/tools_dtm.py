from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Optional
import numpy as np

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
        
        # Sort descending by frequency
        sorted_freqs = dict(sorted(term_freqs.items(), key=lambda x: x[1], reverse=True))
        
        summary = {
            "term_frequencies": sorted_freqs
        }
        
        result_id = session.next_result_id("term_freq")
        summary["result_id"] = result_id
        
        session.store_result("term_frequency_tool", {}, summary)
        
        return summary

    return [build_dtm_tool, term_frequency_tool]
