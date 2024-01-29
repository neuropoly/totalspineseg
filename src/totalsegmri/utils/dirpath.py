from pathlib import Path

class DirPath:
    """
    Get path from argparse and return as Path object.
    
    Args:
        create: Create directory if it doesn't exist
        
    """

    def __init__(self, create=False):
        self.create = create

    def __call__(self, dir_path):
        path = Path(dir_path)
        
        if self.create and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True) 
            except:
                pass
                
        if path.is_dir():
            return path
        else:
            raise argparse.ArgumentTypeError(
                f"readble_dir:{path} is not a valid path")
