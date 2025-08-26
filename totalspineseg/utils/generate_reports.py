import importlib
import os, argparse, textwrap
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import totalspineseg.resources as ressources

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
        '--test-dir', '-t', type=Path, required=True,
        help='The folder where the metrics of the test group are located (required).'
    )
    parser.add_argument(
        '--control-dir', '-c', type=Path, required=True,
        help='The folder where the metrics of the control group are located (required).'
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
    test_path = args.test_dir
    control_path = args.control_dir
    ofolder = args.ofolder
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            test_path = "{test_path}"
            control_path = "{control_path}"
            ofolder = "{ofolder}"
            quiet = {quiet}
        '''))

    generate_reports(
        test_path=test_path,
        control_path=control_path,
        ofolder_path=ofolder,
        quiet=quiet,
    )

def generate_reports(
        test_path,
        control_path,
        ofolder_path,
        quiet=False
    ):
    # Load paths
    test_path = Path(test_path)
    control_path = Path(control_path)
    ofolder_path = Path(ofolder_path)

    # Extract metrics values of the control group
    if not os.path.exists(str(control_path / "all_values.json")):
        all_values = {}
        for subject in os.listdir(control_path):
            if not quiet:
                print(f"Processing subject: {subject}")
            control_sub_folder = control_path / subject

            # Compute metrics subject
            control_data = compute_metrics_subject(control_sub_folder)

            # Gather all values for each metric and structures
            for struc in control_data.keys():
                if struc not in all_values:
                    all_values[struc] = {}
                for struc_name in control_data[struc].keys():
                    if struc_name not in all_values[struc]:
                        all_values[struc][struc_name] = {}
                    for metric in control_data[struc][struc_name].keys():
                        if metric not in all_values[struc][struc_name]:
                            all_values[struc][struc_name][metric] = []

                        # Add subject to all_values
                        subject_value = control_data[struc][struc_name][metric]
                        if subject_value != -1:
                            all_values[struc][struc_name][metric].append(subject_value)
        
        # Align canal and CSF for control group
        all_values = rescale_data(all_values)
    # Create mean dictionary
    mean_dict = {}
    for struc in all_values.keys():
        if struc in ['foramens', 'discs', 'vertebrae']:
            for struc_name in all_values[struc].keys():
                for metric, values in all_values[struc][struc_name].items():
                    mean_value = np.mean(values)
                    if struc not in mean_dict:
                        mean_dict[struc] = {struc_name: {metric: mean_value}}
                    else:
                        if struc_name not in mean_dict[struc]:
                            mean_dict[struc][struc_name] = {metric: mean_value}
                        else:
                            if metric not in mean_dict[struc][struc_name]:
                                mean_dict[struc][struc_name][metric] = mean_value

def compute_metrics_subject(subject_folder):
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
            # Call the compute function to process the data
            merged_data[csv_file] = compute_func(subject_data)
    return merged_data

def compute_canal(subject_data):
    # Convert pandas columns to lists
    canal_dict = {}
    for column in subject_data.columns[2:]:
        if column != 'canal_centroid':
            canal_dict[column] = subject_data[column].tolist()
    return canal_dict

def compute_csf(subject_data):
    # Convert pandas columns to lists
    csf_dict = {}
    for column in subject_data.columns[2:]:
        csf_dict[column] = subject_data[column].tolist()
    return csf_dict

def compute_discs(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def compute_vertebrae(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def compute_foramens(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def rescale_data(all_values):
    '''
    Rescale subject canals and CSF based on discs z coordinates.
    '''
    new_values = copy.deepcopy(all_values)
    for struc in ['canal', 'csf']:
        for struc_name in all_values[struc].keys():
            # Align all metrics for each subject using discs level as references
            disc_levels = all_values[struc][struc_name]['disc_level']
            # Flatten the list of arrays and concatenate all unique values
            all_discs = np.unique(np.concatenate([np.unique(dl) for dl in disc_levels]))
            all_discs = all_discs[~np.isnan(all_discs)]

            # For each subject count slices between discs
            n_subjects = len(disc_levels)
            gap_dict = {}
            for subj_idx in range(n_subjects):
                subj_disc_level = np.array(disc_levels[subj_idx])            
                subj_valid = ~pd.isna(subj_disc_level)
                subj_disc_positions = np.where(subj_valid)[0]
                subj_disc_values = subj_disc_level[subj_valid]

                # If the number of discs doesn't match, skip this subject
                if len(subj_disc_values) < 2:
                    continue
                
                # Create dict with number of slice between discs
                previous_disc = subj_disc_values[0]
                previous_pos = subj_disc_positions[0]
                for pos, disc in zip(subj_disc_positions[1:], subj_disc_values[1:]):
                    if f"{previous_disc}-{disc}" not in gap_dict:
                        gap_dict[f"{previous_disc}-{disc}"] = []
                    gap_dict[f"{previous_disc}-{disc}"].append(pos - previous_pos)
                    previous_disc = disc
                    previous_pos = pos

            # Pick max for each gap between discs in gap_dict
            for k, v in gap_dict.items():
                gap_dict[k] = max(v)

            # Rescale subjects
            for subj_idx in range(n_subjects):
                for metric in all_values[struc][struc_name].keys():
                    if metric in ['slice_nb', 'disc_level']:
                        continue
                    interp_values, slice_interp = rescale_metric(disc_levels[subj_idx], all_values[struc][struc_name][metric][subj_idx], gap_dict)
                    new_values[struc][struc_name][metric][subj_idx] = interp_values
                if 'slice_interp' not in new_values[struc][struc_name]:
                    new_values[struc][struc_name]['slice_interp'] = []
                new_values[struc][struc_name]['slice_interp'].append(slice_interp)
                # Remove slice_nb and disc_level from dict
                new_values[struc][struc_name].pop('slice_nb', None)
                new_values[struc][struc_name].pop('disc_level', None)

            # Store gap_dict in all_values
            new_values[struc][struc_name]['discs_gap'] = gap_dict
    return new_values

def rescale_metric(disc_levels, metric_list, gap_dict):
    '''
    Return rescaled metric values and corresponding slice indices using disc levels and gap information.
    '''
    # Rescale data for each metric
    subj_disc_level = np.array(disc_levels)
    subj_valid = ~pd.isna(subj_disc_level)
    subj_disc_positions = np.where(subj_valid)[0]
    subj_disc_values = subj_disc_level[subj_valid]

    # If the number of discs doesn't match, skip this subject
    if len(subj_disc_values) < 2:
        return [], []

    # Rescale each metric with linear interpolation
    values = np.array(metric_list)
    interp_values = []
    slice_interp = []
    for disc_idx, disc in enumerate(subj_disc_values):
        if disc_idx < len(subj_disc_values) - 1:
            gap = gap_dict[f"{disc}-{subj_disc_values[disc_idx + 1]}"]
            yp = values[subj_disc_positions[disc_idx]:subj_disc_positions[disc_idx+1]]
            xp = np.linspace(0, gap-1, len(yp))
            x = np.linspace(0, gap-1, gap)
            if not -1 in yp:
                interp_func = np.interp(
                    x=x,
                    xp=xp,
                    fp=yp
                )
            else:
                interp_func = np.full_like(x, -1)
            interp_values += interp_func.tolist()

    start_disc_gap = 0
    k = list(gap_dict.keys())[0]
    i = 0
    while k != f"{subj_disc_values[0]}-{subj_disc_values[1]}":
        start_disc_gap += gap_dict[k]
        i += 1
        k = list(gap_dict.keys())[i]
    slice_interp += list(range(start_disc_gap, start_disc_gap + len(interp_values)))
    return interp_values, slice_interp

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

def create_global_figures(subjects_data, all_values, metrics_path, ofolder_path):
    """
    Create global figures from the processed subjects data.

    Parameters:
        subjects_data (dict): A dictionary containing merged metrics data for each subject.
        ofolder_path (Path): Path to the output folder where reports will be saved.
    """
    print("Creating global figures...")
    mean_dict = {}

    # Load totalspineseg ressources path
    ressources_path = importlib.resources.files(ressources)

    # Create discs, vertebrae, foramens figures
    for struc in ['foramens', 'discs', 'vertebrae']:
        # Create a subplot for each subject and overlay a red line corresponding to their value
        for subject in subjects_data.keys():
            struc_names = list(subjects_data[subject][struc].keys())
            metrics = list(subjects_data[subject][struc][struc_names[0]].keys())
            nrows = len(struc_names) + 1
            ncols = len(metrics) + 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
            axes = axes.flatten()
            idx = 0
            for i in range(ncols):
                if i == 0:
                    axes[i].text(0.5, 0.5, "Structure name", fontsize=45, ha='center', va='center')
                elif i == 1:
                    axes[i].text(0.5, 0.5, "Segmentation", fontsize=45, ha='center', va='center')
                else:
                    # Load image 
                    img_path = os.path.join(ressources_path, f'imgs/{struc}_{metrics[i - 2]}.jpg')
                    axes[i].imshow(plt.imread(img_path))
                axes[i].set_axis_off()
                idx += 1
            for struc_name in struc_names:
                axes[idx].text(0.5, 0.5, struc_name, fontsize=45, ha='center', va='center')
                axes[idx].set_axis_off()
                if struc != 'foramens':
                    img_name = f'{struc}_{struc_name}'
                    img = plt.imread(os.path.join(metrics_path, f'{subject}/imgs/{img_name}.png'))
                else:
                    img_name = f'{struc_name}'
                    img_left = plt.imread(os.path.join(metrics_path, f'{subject}/imgs/{img_name}_left.png'))
                    img_right = plt.imread(os.path.join(metrics_path, f'{subject}/imgs/{img_name}_right.png'))

                    # Concatenate images after padding to the maximal shape
                    max_height = max(img_left.shape[0], img_right.shape[0])
                    img_left_padded = np.pad(np.fliplr(img_left), ((0, max_height - img_left.shape[0]), (0, 0)), mode='constant')
                    img_right_padded = np.pad(img_right, ((0, max_height - img_right.shape[0]), (0, 0)), mode='constant')
                    img = np.concatenate((img_right_padded, img_left_padded), axis=1)

                axes[idx+1].imshow(img)
                axes[idx+1].set_axis_off()
                idx += 2
                for metric in metrics:
                    ax = axes[idx]
                    subject_value = subjects_data[subject][struc][struc_name][metric]
                    values = all_values[struc][struc_name][metric]

                    # Create mean dictionary
                    mean_value = np.mean(values)
                    if struc not in mean_dict:
                        mean_dict[struc] = {struc_name: {metric: mean_value}}
                    else:
                        if struc_name not in mean_dict[struc]:
                            mean_dict[struc][struc_name] = {metric: mean_value}
                        else:
                            if metric not in mean_dict[struc][struc_name]:
                                mean_dict[struc][struc_name][metric] = mean_value

                    # Plot metric for subject
                    # If subject_value >= mean_value, make the violin plot transparent
                    if subject_value >= mean_value or subject_value == -1:
                        sns.violinplot(x=values, ax=ax, cut=0, bw_method=0.7, color='gray', alpha=0.2)
                    else:
                        sns.violinplot(x=values, ax=ax, cut=0, bw_method=0.7)
                    ax.tick_params(axis='x', rotation=45, labelsize=12)
                    if subject_value != -1:
                        axes[idx].axvline(x=subject_value, color='red', linestyle='--')
                    fig.tight_layout()
                    idx += 1
                
            plt.savefig(str(ofolder_path / subject / f"compared_{struc}.png"))


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