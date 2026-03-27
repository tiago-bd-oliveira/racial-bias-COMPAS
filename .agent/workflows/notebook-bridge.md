---
description: Sets up a temporary Python bridge to edit Jupyter Notebooks without breaking JSON.
---

### Step 1: Initialize the Bridge Script // turbo
Create a file at `/tmp/nb_bridge.py` with this logic:
```python
import nbformat as nbf
import sys, os

def modify_nb(path, content, cell_type='code'):
    if not os.path.exists(path):
        nb = nbf.v4.new_notebook()
    else:
        with open(path, 'r', encoding='utf-8') as f:
            nb = nbf.read(f, as_version=4)
    
    new_cell = nbf.v4.new_code_cell(content) if cell_type == 'code' else nbf.v4.new_markdown_cell(content)
    nb.cells.append(new_cell)
    
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def replace_in_nb(path, search_content, replacement_content):
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
    
    modified = False
    for cell in nb.cells:
        source = "".join(cell.source) if isinstance(cell.source, list) else cell.source
        if search_content in source:
            cell.source = source.replace(search_content, replacement_content)
            modified = True
    
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)

def query_nb(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
    
    print(f"Notebook Summary: {path}")
    print(f"{'Index':<5} | {'Type':<10} | {'Content Preview'}")
    print("-" * 60)
    for i, cell in enumerate(nb.cells):
        source = "".join(cell.source) if isinstance(cell.source, list) else cell.source
        preview = source.replace('\n', ' ')[:60].strip()
        if len(source) > 60: preview += "..."
        print(f"{i:<5} | {cell.cell_type:<10} | {preview}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 /tmp/nb_bridge.py <mode: append|replace|query> <path> [args...]")
        sys.exit(1)
    
    mode = sys.argv[1]
    path = sys.argv[2]
    
    if mode == "append":
        modify_nb(path, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else 'code')
    elif mode == "replace":
        replace_in_nb(path, sys.argv[3], sys.argv[4])
    elif mode == "query":
        query_nb(path)
```

---

## Step 2: Add a Supporting Rule (The "Toggle")
While the workflow creates the tool, a **Rule** tells the agent *how* to use it. This acts as the persistent "brain" for the session.

1.  Open the **Customizations** panel (the `...` menu in the top right of the Agent Manager).
2.  Go to **Rules** > **+ Workspace**.
3.  Name it `notebook-expert` and set the trigger to **Manual**.
4.  Paste this instruction:
    > "When working with .ipynb files, do not edit the JSON directly. Always use the bridge script located at `/tmp/nb_bridge.py`. 
    > **Usage Strategy:** 
    > 1. First, use `python3 /tmp/nb_bridge.py query [file]` to list cells and identify where to make changes.
    > 2. Use `replace` to edit existing code or `append` to add new analysis.
    > If the script is missing, ask the user to run `/notebook-bridge` first."

---

## Step 3: Using it in a Chat Session
Now that you've configured it, your interaction becomes a simple two-step automated process:

1.  **Activate the Tool:** Type `/notebook-bridge` in the chat.
2.  **Give the Command:** 
    -   **Query:** *"What's in 'analysis.ipynb'?"* (Agent will query first)
    -   **Append:** *"Add a cell to 'analysis.ipynb' that imports pandas."*
    -   **Replace:** *"Change 'old_code' to 'new_code' in 'model.ipynb'."*

The agent will now use the terminal to run:
`python3 /tmp/nb_bridge.py [mode] [file] [args...]`