from langchain_core.tools import tool
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def _compute_correlation(session, target_class, method="pearson"):
    """Helper to compute correlation between terms and a target class."""
    if session.feature_matrix is None or session.feature_names is None:
        return {"error": "No document-term matrix built yet. Run build_dtm_tool first."}
    
    if session.labels is None:
        return {"error": "No category labels found in session.labels. Run build_dtm_tool first."}
    
    valid_categories = list(session.categories) if session.categories is not None else list(np.unique(session.labels))
    if target_class not in valid_categories:
        return {"error": f"Target class '{target_class}' not found in categories: {valid_categories}"}
    
    y = (session.labels == target_class).astype(int)
    dense_matrix = session.feature_matrix.toarray()
    terms = session.feature_names
    
    corrs = []
    for col_idx in range(dense_matrix.shape[1]):
        col = dense_matrix[:, col_idx]
        if np.std(col) == 0 or np.std(y) == 0:
            r = 0.0
        else:
            if method == "pearson":
                r, _ = pearsonr(col, y)
            else:
                r, _ = spearmanr(col, y)
            if np.isnan(r):
                r = 0.0
        corrs.append(float(r))
    
    return terms, corrs

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Flags near-constant terms by variance for removal from the global DTM.

        Args:
            threshold: Variance threshold below which terms are flagged for removal.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built yet. Run build_dtm_tool first."}
        
        dense_matrix = session.feature_matrix.toarray()
        variances = np.var(dense_matrix, axis=0)
        
        terms = session.feature_names
        
        report_df = pd.DataFrame({
            "term": terms,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)
        
        kept_terms = []
        dropped_terms = []
        
        for _, row in report_df.iterrows():
            term = row["term"]
            var = row["variance"]
            if var >= threshold:
                kept_terms.append(term)
            else:
                dropped_terms.append(term)
                
        summary = {
            "threshold": threshold,
            "kept_count": len(kept_terms),
            "dropped_count": len(dropped_terms),
            "kept_terms": kept_terms,
            "dropped_terms": dropped_terms
        }
        
        result_id = session.next_result_id("var_filter")
        summary["result_id"] = result_id
        
        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=report_df
        )
        
        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Computes Pearson correlation between each term's count and a target class label.

        Args:
            target_class: The target category label for one-vs-rest binary correlation.
            threshold: Absolute correlation threshold below which terms are flagged for removal.
        """
        res = _compute_correlation(session, target_class, method="pearson")
        if isinstance(res, dict): return res
        terms, pearson_rs = res
            
        report_df = pd.DataFrame({
            "term": terms,
            "pearson_r": pearson_rs
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)
        
        kept_terms = []
        dropped_terms = []
        
        for _, row in report_df.iterrows():
            term = row["term"]
            r = row["pearson_r"]
            if abs(r) >= threshold:
                kept_terms.append(term)
            else:
                dropped_terms.append(term)
                
        summary = {
            "target_class": target_class,
            "threshold": threshold,
            "kept_count": len(kept_terms),
            "dropped_count": len(dropped_terms),
            "kept_terms": kept_terms,
            "dropped_terms": dropped_terms
        }
        
        result_id = session.next_result_id("pearson_filter")
        summary["result_id"] = result_id
        
        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df
        )
        
        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Computes Spearman correlation between each term's count and a target class label.

        Args:
            target_class: The target category label for one-vs-rest binary correlation.
            threshold: Absolute correlation threshold below which terms are flagged for removal.
        """
        res = _compute_correlation(session, target_class, method="spearman")
        if isinstance(res, dict): return res
        terms, spearman_rs = res
            
        report_df = pd.DataFrame({
            "term": terms,
            "spearman_r": spearman_rs
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)
        
        kept_terms = []
        dropped_terms = []
        
        for _, row in report_df.iterrows():
            term = row["term"]
            r = row["spearman_r"]
            if abs(r) >= threshold:
                kept_terms.append(term)
            else:
                dropped_terms.append(term)
                
        summary = {
            "target_class": target_class,
            "threshold": threshold,
            "kept_count": len(kept_terms),
            "dropped_count": len(dropped_terms),
            "kept_terms": kept_terms,
            "dropped_terms": dropped_terms
        }
        
        result_id = session.next_result_id("spearman_filter")
        summary["result_id"] = result_id
        
        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df
        )
        
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
