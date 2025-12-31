import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

def run_notebook(notebook_path):
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found.")
        return False

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        print(f"Executing {notebook_path}...")
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("Execution finished successfully.")
        return True
    except Exception as e:
        print(f"Error executing the notebook: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_notebook(sys.argv[1])
    else:
        print("Usage: python scripts/run_notebook_v2.py <notebook_path>")
