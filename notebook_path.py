import os 
import sys 

def NotebookPath(global_path: str, local_path: str):
    global_path = os.path.abspath(global_path)
    full_path = os.path.join(global_path, local_path)
    
    for p in [global_path, full_path]:
        if p not in sys.path:
            sys.path.insert(0, p)
    
