import os
import tempfile
import numpy as np
import pandas as pd
from typing import Literal, Optional
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> dict:
        """Mines frequent patterns for a single category using PAMI algorithms.

        Filters session.dataframe to the requested category, creates a category-local DTM
        with English stopwords removed, applies vocabulary filtering, converts to a transactional
        database, and runs the requested PAMI algorithm (FPGrowth, Top-K FAE, or MaxFPGrowth).

        Args:
            category_name: The category name to filter and mine documents for.
            filtering_method: Vocabulary filtering method ("variance", "tfidf", or "term_frequency").
            algorithm: Mining algorithm to run ("fpgrowth", "topk", or "maxfpgrowth").
            min_sup: Minimum support threshold (required for "fpgrowth" and "maxfpgrowth").
            k: Top-k pattern count limit (required for "topk").
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}

        df = session.dataframe
        if "category_name" not in df.columns or "text" not in df.columns:
            return {"error": "DataFrame missing 'category_name' or 'text' column."}

        # Filter to requested category
        cat_df = df[df["category_name"] == category_name]
        if cat_df.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        # Validate inputs
        if algorithm in ["fpgrowth", "maxfpgrowth"] and min_sup is None:
            return {"error": f"Parameter 'min_sup' is required when algorithm='{algorithm}'."}
        if algorithm == "topk" and k is None:
            return {"error": "Parameter 'k' is required when algorithm='topk'."}

        # 1. Fit category-local CountVectorizer
        texts = cat_df["text"].fillna("").tolist()
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts).toarray()
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            # e.g., empty vocabulary after stopword removal
            summary = {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "total_patterns": 0,
                "patterns": [],
                "note": "no terms survived vectorization"
            }
            result_id = session.next_result_id("mine_patterns")
            summary["result_id"] = result_id
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        if dtm.shape[1] == 0:
            summary = {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "total_patterns": 0,
                "patterns": [],
                "note": "no terms survived vectorization"
            }
            result_id = session.next_result_id("mine_patterns")
            summary["result_id"] = result_id
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 2. Apply filtering method
        # Middle 90% means at or above 5th percentile and at or below 95th percentile (inclusive: p5 <= s <= p95)
        # Top 80% means at or above 20th percentile (inclusive: s >= p20)
        if filtering_method == "variance":
            scores = np.var(dtm, axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            kept_indices = [i for i, s in enumerate(scores) if p5 <= s <= p95]
        elif filtering_method == "term_frequency":
            scores = np.sum(dtm, axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            kept_indices = [i for i, s in enumerate(scores) if p5 <= s <= p95]
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_mat = tfidf_transformer.fit_transform(dtm).toarray()
            scores = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(scores, 20)
            kept_indices = [i for i, s in enumerate(scores) if s >= p20]
        else:
            return {"error": f"Unknown filtering method '{filtering_method}'."}

        # 3. Check if terms survived
        if len(kept_indices) == 0:
            summary = {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "total_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            result_id = session.next_result_id("mine_patterns")
            summary["result_id"] = result_id
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        kept_dtm = dtm[:, kept_indices]
        kept_terms = [feature_names[i] for i in kept_indices]

        # 4. Transactional DB conversion & mining via PAMI
        # Filter out rows with zero items
        row_sums = kept_dtm.sum(axis=1)
        valid_rows = kept_dtm[row_sums > 0]

        if valid_rows.shape[0] == 0:
            summary = {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "total_patterns": 0,
                "patterns": [],
                "note": "no transactions found"
            }
            result_id = session.next_result_id("mine_patterns")
            summary["result_id"] = result_id
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        df_trans = pd.DataFrame(valid_rows, columns=kept_terms)

        patterns = []
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "db.txt")
            output_file = os.path.join(tmpdir, "output.txt")

            try:
                from PAMI.extras.convert import DF2DB
                df2db = DF2DB.DF2DB(df_trans)
                df2db.convert2TransactionalDatabase(db_file, ">", 0)

                with open(db_file, "rb") as f:
                    raw = f.read()

                with open(db_file, "w", encoding="utf-8") as f:
                    f.write(raw.decode("latin-1"))
            except Exception as e:
                return {"error": f"Failed to convert DataFrame to transactional DB: {str(e)}"}

            try:
                if algorithm == "fpgrowth":
                    from PAMI.frequentPattern.basic import FPGrowth
                    obj = FPGrowth.FPGrowth(db_file, minSup=min_sup)
                elif algorithm == "topk":
                    from PAMI.frequentPattern.topk import FAE
                    obj = FAE.FAE(db_file, k=k)
                elif algorithm == "maxfpgrowth":
                    from PAMI.frequentPattern.maximal import MaxFPGrowth
                    obj = MaxFPGrowth.MaxFPGrowth(db_file, minSup=min_sup)

                obj.mine()
                obj.save(output_file)

                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split(":")
                            if len(parts) >= 2:
                                items = parts[0].strip().split("\t")
                                try:
                                    support = float(parts[1].strip())
                                except ValueError:
                                    support = parts[1].strip()
                                patterns.append({"pattern": items, "support": support})
            except Exception as e:
                return {"error": f"PAMI mining failed: {str(e)}"}

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        summary = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "total_patterns": len(patterns),
            "patterns": patterns[:20],
            "artifact_key": artifact_key
        }

        result_id = session.next_result_id("mine_patterns")
        summary["result_id"] = result_id

        session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)

        return summary

    return [mine_patterns_tool]
