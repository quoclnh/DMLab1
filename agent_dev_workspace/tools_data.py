import os
from langchain_core.tools import tool

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

    return [list_files_tool, inspect_data_tool]
