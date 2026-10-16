import os
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from typing import Optional

def make_tools(session):
    @tool
    def list_files_tool(subdirectory: str = ".") -> dict:
        """Lists files and folders in the specified repository subdirectory.
        
        Useful for discovering available dataset files before loading them into the session.

        Args:
            subdirectory: The directory path relative to the repo root to list, e.g. "." or "newdataset".
        """
        target_path = subdirectory
        
        if not os.path.exists(target_path):
            return {"error": f"Subdirectory '{subdirectory}' does not exist.", "directories": [], "files": []}
        
        try:
            with os.scandir(target_path) as entries:
                directories = []
                files = []
                for entry in entries:
                    if entry.is_dir():
                        directories.append(entry.name)
                    elif entry.is_file():
                        files.append(entry.name)
        except Exception as e:
            return {"error": str(e), "directories": [], "files": []}
        
        directories.sort()
        files.sort()
        
        summary = {
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }
        
        result_id = session.next_result_id("list_files")
        summary["result_id"] = result_id
        
        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> dict:
        """Peeks at the first n rows of the working DataFrame in session.dataframe.

        Args:
            n: Number of rows to inspect (must be a positive integer).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}
        
        if not isinstance(n, int) or n <= 0:
            return {"error": "Parameter 'n' must be a positive integer."}
        
        df = session.dataframe
        head_df = df.head(n)
        records = head_df.to_dict(orient="records")
        
        summary = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "n_inspected": len(records),
            "rows": records
        }
        
        result_id = session.next_result_id("inspect")
        summary["result_id"] = result_id
        
        session.store_result("inspect_data_tool", {"n": n}, summary)
        
        return summary

    @tool
    def check_missing_tool() -> dict:
        """Checks session.dataframe for missing or empty text values.

        Returns a summary of how many rows have missing or blank text.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}
        
        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "DataFrame missing 'text' column."}
        
        missing_mask = df["text"].isna() | df["text"].astype(str).str.strip().eq("")
        n_missing = int(missing_mask.sum())
        
        summary = {
            "total_rows": len(df),
            "n_missing": n_missing
        }
        
        result_id = session.next_result_id("missing")
        summary["result_id"] = result_id
        
        session.store_result("check_missing_tool", {}, summary)
        
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Checks for (and optionally drops) duplicate documents in session.dataframe.

        When dropping, uses keep=False to remove all copies of duplicated rows.

        Args:
            drop: Whether to drop duplicate rows in place from session.dataframe.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}
        
        df = session.dataframe
        original_rows = len(df)
        
        dupes_mask = df.duplicated(subset=["text"], keep=False)
        n_duplicates = int(dupes_mask.sum())
        
        n_dropped = 0
        if drop and n_duplicates > 0:
            before_len = len(df)
            df.drop_duplicates(subset=["text"], keep=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            n_dropped = before_len - len(df)
        
        summary = {
            "total_rows": len(df),
            "n_duplicates": n_duplicates,
            "n_dropped": n_dropped,
            "dropped": drop
        }
        
        result_id = session.next_result_id("duplicates")
        summary["result_id"] = result_id
        
        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        
        return summary

    @tool
    def sample_data_tool(n: int = 5, random_state: Optional[int] = None) -> dict:
        """Subsamples the working DataFrame (session.dataframe).

        Args:
            n: Number of rows to sample.
            random_state: Optional random seed for reproducible sampling.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}
        
        df = session.dataframe
        if not isinstance(n, int) or n <= 0:
            return {"error": "Parameter 'n' must be a positive integer."}
        
        if n > len(df):
            return {"error": f"Requested sample size n={n} exceeds total rows in dataset ({len(df)})."}
        
        sampled_df = df.sample(n=n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()
        records = sampled_df.to_dict(orient="records")
        
        summary = {
            "n_sampled": len(records),
            "sampled_indices": sampled_indices,
            "rows": records
        }
        
        result_id = session.next_result_id("sample")
        summary["result_id"] = result_id
        
        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        
        return summary

    @tool
    def tokenize_tool() -> dict:
        """Tokenizes each document's text in session.dataframe into unigrams.

        Uses nltk.word_tokenize on lowercased text and stores the token lists
        in session.dataframe["unigrams"].
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}
        
        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "DataFrame missing 'text' column."}
        
        for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab']:
            try:
                nltk.data.find(resource)
            except LookupError:
                res_name = resource.split('/')[-1]
                try:
                    nltk.download(res_name, quiet=True)
                    nltk.data.find(resource)
                except Exception as e:
                    return {"error": f"Required NLTK resource '{res_name}' is unavailable: {str(e)}"}
        
        token_lists = []
        for text in df["text"]:
            if pd.isna(text) or not str(text).strip():
                token_lists.append([])
            else:
                tokens = nltk.word_tokenize(str(text).lower())
                token_lists.append(tokens)
                
        df["unigrams"] = token_lists
        
        summary = {
            "n_documents": len(df),
            "total_tokens": sum(len(t) for t in token_lists)
        }
        
        result_id = session.next_result_id("tokenize")
        summary["result_id"] = result_id
        
        session.store_result("tokenize_tool", {}, summary)
        
        return summary

    @tool
    def describe_data_tool() -> dict:
        """Computes descriptive statistics on document text lengths (character counts)

        overall and per category, and generates a box plot set on session.pending_figure.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}
        
        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "DataFrame missing 'text' column."}
        
        df["text_length"] = df["text"].fillna("").apply(len)
        
        overall_stats = df["text_length"].describe().to_dict()
        
        per_category_stats = {}
        if "category_name" in df.columns:
            for cat, group in df.groupby("category_name"):
                per_category_stats[str(cat)] = group["text_length"].describe().to_dict()
                
        # Generate box plot
        fig, ax = plt.subplots(figsize=(8, 6))
        if "category_name" in df.columns:
            sns.boxplot(x="category_name", y="text_length", data=df, ax=ax)
        else:
            sns.boxplot(y="text_length", data=df, ax=ax)
        ax.set_title("Document Text Length by Category")
        ax.set_xlabel("Category" if "category_name" in df.columns else "")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()
        
        session.pending_figure = fig
        
        summary = {
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats
        }
        
        result_id = session.next_result_id("describe")
        summary["result_id"] = result_id
        
        session.store_result("describe_data_tool", {}, summary)
        
        return summary

    return [
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        tokenize_tool,
        describe_data_tool
    ]
