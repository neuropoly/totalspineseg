from pathlib import Path
import nnunetv2
import textwrap

def main():
    # Locate the path to the nnunetv2 library
    nnunetv2_path = Path(nnunetv2.__path__[0])

    # Define the target file
    target_file = nnunetv2_path / "training" / "nnUNetTrainer" / "nnUNetTrainer_16000epochs.py"

    # Ensure the target file exists and create it if it is not
    if not target_file.exists():
        # Add the class definition to the target file
        target_file.write_text(textwrap.dedent("""
            import torch
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

            class nnUNetTrainer_16000epochs(nnUNetTrainer):
                def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                            device: torch.device = torch.device('cuda')):
                    super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
                    self.num_epochs = 16000
        """).strip())

if __name__ == "__main__":
    main()