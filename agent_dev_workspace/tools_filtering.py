from langchain_core.tools import tool
import pandas as pd
import numpy as np

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Flags near-constant terms by variance for removal from the global DTM.

        Args:
            threshold: Variance threshold below which terms are flagged for removal.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix built yet. Run build_dtm_tool first."}
        
        # Convert sparse matrix to dense array for variance calculation
        dense_matrix = session.feature_matrix.toarray()
        variances = np.var(dense_matrix, axis=0)
        
        terms = session.feature_names
        
        # Build DataFrame report: term as first column, variance as second column, sorted by variance descending
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

    return [variance_filter_tool]
