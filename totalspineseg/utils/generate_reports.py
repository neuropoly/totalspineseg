import sys, os, argparse, textwrap
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes the outputs of totalspineseg/utils/measure_seg.py to generate a reports.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            totalspineseg_generate_reports -m metrics_folder -o reports
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--metrics-dir', '-m', type=Path, required=True,
        help='The folder where computed metrics are located (required).'
    )
    parser.add_argument(
        '--ofolder', '-o', type=Path, required=True,
        help='The folder where reports will be saved (required).'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    metrics_path = args.metrics_dir
    ofolder = args.ofolder
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            metrics_path = "{metrics_path}"
            ofolder = "{ofolder}"
            quiet = {quiet}
        '''))

    generate_reports(
        metrics_path=metrics_path,
        ofolder_path=ofolder,
        quiet=quiet,
    )

def generate_reports(
        metrics_path,
        ofolder_path,
        quiet=False
    ):
    # Load paths
    metrics_path = Path(metrics_path)
    ofolder_path = Path(ofolder_path)

    # Loop across subject folders under metrics_path
    subjects_data = {}
    for subject in os.listdir(metrics_path):
        subject_folder = metrics_path / subject

        # Compute metrics subject
        ofolder_subject = ofolder_path / subject / 'plots'
        ofolder_subject.mkdir(parents=True, exist_ok=True)
        subjects_data[subject] = compute_metrics_subject(subject_folder, ofolder_subject, quiet)

    # Create global figures
    create_global_figures(subjects_data, ofolder_path)

def compute_metrics_subject(subject_folder, ofolder_path, quiet=False):
    """
    Compute metrics for a single subject and return merged_data dict for global figures.

    Parameters:
        subject_folder (Path): Path to the subject's metrics folder.
        ofolder_path (Path): Path to the output folder where reports will be saved.
        quiet (bool, optional): If True, suppresses output messages. Defaults to False.

    Returns:
        dict: A dictionary containing merged metrics data for the subject.
    """
    merged_data = {}

    # List of expected CSV files
    csv_files = {
        "canal":compute_canal, 
        "csf":compute_csf, 
        "discs":compute_discs, 
        "foramens":compute_foramens, 
        "vertebrae":compute_vertebrae
    }

    # Load each CSV if it exists
    for csv_file, compute_func in csv_files.items():
        csv_path = subject_folder / 'csv' / f"{csv_file}.csv"
        if csv_path.exists():
            subject_data = pd.read_csv(str(csv_path))
            if not quiet:
                print(f"Processing {csv_file} for subject {subject_folder.name}")
            # Call the compute function to process the data
            merged_data[csv_file] = compute_func(subject_data, ofolder_path)

    return merged_data

def compute_canal(subject_data, ofolder_path):
    # Create plots for all the metrics
    plot_metrics(subject_data, ofolder_path, structure="canal")
    return {'canal': None}

def compute_csf(subject_data, ofolder_path):
    # Create plots for all the metrics
    plot_metrics(subject_data, ofolder_path, structure="csf")
    return {'csf': None}

def compute_discs(subject_data, ofolder_path):
    # Plot bar plot for intensity profile
    plot_intensity_profile(subject_data, ofolder_path, structure="disc")

    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def compute_vertebrae(subject_data, ofolder_path):
     # Plot bar plot for intensity profile
    plot_intensity_profile(subject_data, ofolder_path, structure="vertebrae")

    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def compute_foramens(subject_data, ofolder_path):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def plot_intensity_profile(subject_data, ofolder_path, structure):
    for struc in subject_data.name:
        struc_data = subject_data[subject_data['name'] == struc]
        struc_idx = struc_data.index[0]
        plt.figure()
        intensity_profile = convert_str_to_list(struc_data['intensity_profile'][struc_idx])
        plt.bar(range(len(intensity_profile)), intensity_profile)
        plt.title(f"{structure} {struc} intensity profile")
        plt.xlabel("Z index")
        plt.ylabel("Intensity")
        plt.savefig(str(ofolder_path / f"{structure}_{struc}_intensity_profile.png"))
        plt.close()

def plot_metrics(subject_data, ofolder_path, structure):
    for metric in subject_data.columns[3:]:
        plt.figure()
        plt.plot(subject_data['slice_nb'], subject_data[metric])
        plt.title(f"{structure} {metric}")
        plt.xlabel("Slice number")
        plt.ylabel(metric)
        plt.savefig(str(ofolder_path / f"{structure}_{metric}.png"))
        plt.close()

def create_dict_from_subject_data(subject_data):
    """
    Create a dictionary from the subject data DataFrame.

    Parameters:
        subject_data (pd.DataFrame): The subject data DataFrame.

    Returns:
        dict: A dictionary with structure names as keys and DataFrames as values.
    """
    subject_dict = {}
    for struc in subject_data.name:
        struc_dict = {}
        struc_data = subject_data[subject_data['name'] == struc]
        struc_idx = struc_data.index[0]
        for column in struc_data.columns[2:]:
            if column == 'intensity_profile':
                struc_dict['intensity_mean'] = np.mean(convert_str_to_list(struc_data[column][struc_idx]))
            else:
                if column != 'center':
                    struc_dict[column] = struc_data[column][struc_idx]
        subject_dict[struc] = struc_dict
    return subject_dict

def create_global_figures(subjects_data, ofolder_path):
    """
    Create global figures from the processed subjects data.

    Parameters:
        subjects_data (dict): A dictionary containing merged metrics data for each subject.
        ofolder_path (Path): Path to the output folder where reports will be saved.
    """
    print("Creating global figures...")
    mean_dict = {}

    def next_even(N):
        return N if N % 2 == 0 else N + 1

    # Create discs, vertebrae, foramens figures
    for struc in ['discs', 'vertebrae', 'foramens']:
        for struc_name in subjects_data[list(subjects_data.keys())[0]][struc].keys():
            metrics = list(subjects_data[list(subjects_data.keys())[0]][struc][struc_name].keys())
            N = len(metrics)
            nrows = 2
            ncols = next_even(N) // nrows
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
            axes = axes.flatten() if N > 1 else [axes]

            for idx, metric in enumerate(metrics):
                values = []
                for subject in subjects_data.keys():
                    values.append(subjects_data[subject][struc][struc_name][metric])

                # Create mean dictionary
                mean_value = np.mean(values)
                if struc not in mean_dict:
                    mean_dict[struc] = {struc_name: {metric: mean_value}}
                else:
                    if struc_name not in mean_dict[struc]:
                        mean_dict[struc][struc_name] = {metric: mean_value}
                    else:
                        mean_dict[struc][struc_name][metric] = mean_value
                
                # Plot global distribution
                ax = axes[idx]
                sns.violinplot(x=values, ax=ax, cut=0, bw=0.2)
                ax.set_title(f"{struc_name} {metric} distribution")
                ax.set_xlabel(metric)
                ax.tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for j in range(idx + 1, nrows * ncols):
                axes[j].set_visible(False)
            
            # Save the base violin plot
            fig.tight_layout()
            plt.savefig(str(ofolder_path / f"global_{struc}_{struc_name}.png"))

            # Overlay a red line for each subject and save, then remove
            for subject in subjects_data.keys():
                lines = []
                for idx, metric in enumerate(metrics):
                    subject_value = subjects_data[subject][struc][struc_name][metric]
                    line = axes[idx].axvline(x=subject_value, color='red', linestyle='--')
                    lines.append(line)
                    fig.tight_layout()
                plt.savefig(str(ofolder_path / subject / f"compared_{struc}_{struc_name}.png"))
                for line in lines:
                    line.remove()


def convert_str_to_list(string):
    return [float(item.strip()) for item in string.split()[1:-1]]

if __name__ == "__main__":
    metrics_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/out/metrics_output'
    ofolder = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/code/totalspineseg/test'
    quiet = False
    generate_reports(
        metrics_path=metrics_path,
        ofolder_path=ofolder,
        quiet=quiet,
    )