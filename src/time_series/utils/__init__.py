import yaml
from pathlib import Path

def read_yaml(yaml_path: Path) -> dict:
    
    with open(yaml_path, mode='r') as f:
        data= yaml.safe_load(f)
        return data