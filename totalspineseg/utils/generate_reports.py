import importlib
import os, argparse, textwrap
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import copy
import totalspineseg.resources as ressources

def main():
    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes the outputs of totalspineseg/utils/measure_seg.py to generate a reports.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            totalspineseg_generate_reports -t test_group_folder -c control_group_folder -o reports
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
                for struc_name in control_data[struc].keys():
                    for metric in control_data[struc][struc_name].keys():
                        if metric != 'intensity':
                            # Add subject to all_values
                            subject_value = control_data[struc][struc_name][metric]
                            if subject_value != -1:
                                if struc not in all_values:
                                    all_values[struc] = {}
                                if struc_name not in all_values[struc]:
                                    all_values[struc][struc_name] = {}
                                if metric not in all_values[struc][struc_name]:
                                    all_values[struc][struc_name][metric] = []
                                all_values[struc][struc_name][metric].append(subject_value)
        
        # Align canal and CSF for control group
        all_values = rescale_canal(all_values)

        # TODO : save all values

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

    # Convert all_values to dataframe
    all_values_df = convert_to_df(all_values)
    
    # Create global figures for test data subjects
    for subject in os.listdir(test_path):
        if not quiet:
            print(f"Creating global figures for subject: {subject}")
        test_sub_folder = test_path / subject

        # Load subject data
        subject_data = compute_metrics_subject(test_sub_folder)

        # Rescale canal and CSF metrics
        interp_data = copy.deepcopy(subject_data)
        for struc in ['canal', 'csf']:
            for struc_name in subject_data[struc].keys():
                for metric in subject_data[struc][struc_name].keys():
                    if metric in ['slice_nb', 'disc_level']:
                        continue
                    interp_values, slice_interp = rescale_with_discs(subject_data[struc][struc_name]['disc_level'], subject_data[struc][struc_name][metric], all_values[struc][struc_name]['discs_gap'])
                    interp_data[struc][struc_name][metric] = interp_values
                interp_data[struc][struc_name]['slice_interp'] = slice_interp
                # remove slice_nb and disc_level from dict
                interp_data[struc][struc_name].pop('slice_nb', None)
                interp_data[struc][struc_name].pop('disc_level', None)

        # Create paths
        imgs_path = test_path / f'{subject}/imgs'
        ofolder_subject = ofolder_path / subject
        discs_gap = all_values[struc][struc_name]['discs_gap']
        ofolder_subject.mkdir(parents=True, exist_ok=True)
        create_global_figures(interp_data, all_values_df, discs_gap, mean_dict, imgs_path, ofolder_subject)

def convert_to_df(all_values):
    new_values = copy.deepcopy(all_values)
    for struc in all_values.keys():
        for struc_name in all_values[struc].keys():
            # Convert dict to dataframe with keys as columns and lines as subjects
            # Prepare a dictionary where each key is a metric and each value is a list of values for all subjects
            for i, metric in enumerate(all_values[struc][struc_name].keys()):
                if metric not in ['discs_gap', 'slice_interp']:
                    data = {'subjects' : [], 'values' : [], 'slice_interp' : []}
                    for j, subject_value in enumerate(all_values[struc][struc_name][metric]):
                        if isinstance(subject_value, list):
                            for value, slice_interp in zip(subject_value, all_values[struc][struc_name]['slice_interp'][j]):
                                data['values'].append(value)
                                data['slice_interp'].append(slice_interp)
                                data['subjects'].append(f'subject_{j}')
                        else:
                            data['values'].append(subject_value)
                            data['subjects'].append(f'subject_{j}')
                df = pd.DataFrame.from_dict(
                    data,
                    orient='index'
                ).transpose()
                new_values[struc][struc_name][metric] = df
    return new_values

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
        "canal":process_canal, 
        "csf":process_csf, 
        "discs":process_discs, 
        "foramens":process_foramens, 
        "vertebrae":process_vertebrae
    }

    # Load each CSV if it exists
    for csv_file, process_func in csv_files.items():
        csv_path = subject_folder / 'csv' / f"{csv_file}.csv"
        if csv_path.exists():
            subject_data = pd.read_csv(str(csv_path))
            # Call the compute function to process the data
            merged_data[csv_file] = process_func(subject_data)
    return merged_data

def process_canal(subject_data):
    # Convert pandas columns to lists
    canal_dict = {'canal': {}, 'spinalcord': {}, 'spinalcord/canal': {}}
    for column in subject_data.columns[2:]:
        if column not in ['canal_centroid', 'angle_AP', 'angle_RL', 'length']:
            if not 'canal' in column:
                canal_dict['spinalcord'][column.replace('_spinalcord','')] = subject_data[column].tolist()
            if not 'spinalcord' in column:
                canal_dict['canal'][column.replace('_canal','')] = subject_data[column].tolist()
    
    # Create spinalcord/canal quotient
    for key in canal_dict['spinalcord'].keys():
        if not key in ['slice_nb', 'disc_level']:
            canal_dict['spinalcord/canal'][key] = []
            for i in range(len(canal_dict['spinalcord'][key])):
                canal_value = canal_dict['canal'][key][i]
                spinalcord_value = canal_dict['spinalcord'][key][i]
                if canal_value != 0 and canal_value != -1 and spinalcord_value != -1:
                    canal_dict['spinalcord/canal'][key].append(spinalcord_value / canal_value)
                else:
                    canal_dict['spinalcord/canal'][key].append(-1)
        else:
            canal_dict['spinalcord/canal'][key] = canal_dict['spinalcord'][key]
    return canal_dict

def process_csf(subject_data):
    # Convert pandas columns to lists
    csf_dict = {'csf': {}}
    for column in subject_data.columns[2:]:
        csf_dict['csf'][column] = subject_data[column].tolist()
    return csf_dict

def process_discs(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def process_vertebrae(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data, intensity_profile=False)
    return subject_dict

def process_foramens(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def rescale_canal(all_values):
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
                    interp_values, slice_interp = rescale_with_discs(disc_levels[subj_idx], all_values[struc][struc_name][metric][subj_idx], gap_dict)
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

def rescale_with_discs(disc_levels, metric_list, gap_dict):
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

def create_dict_from_subject_data(subject_data, intensity_profile=True):
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
                if intensity_profile:
                    struc_dict['intensity'] = convert_str_to_list(struc_data[column][struc_idx])
            else:
                if column != 'center':
                    struc_dict[column] = struc_data[column][struc_idx]
        subject_dict[struc] = struc_dict
    return subject_dict

def create_global_figures(subject_data, all_values_df, discs_gap, mean_dict, imgs_path, ofolder_path):
    """
    Create global figures from the processed subjects data.

    Parameters:
        subject_data (dict): A dictionary containing the subject data.
        all_values (dict): A dictionary containing all processed metrics data for each control subject.
        imgs_path (Path): Path to the folder containing the subject images.
        ofolder_path (Path): Path to the output folder where reports will be saved.
    """

    # Load totalspineseg ressources path
    ressources_path = importlib.resources.files(ressources)
    with open(os.path.join(ressources_path, 'labels_maps/levels_maps.json'), 'r') as f:
        mapping = json.load(f)
        rev_mapping = {v: k for k, v in mapping.items()}

    # Create discs, vertebrae, foramens figures
    for struc in ['canal', 'csf']:
        # Create a subplot for each subject and overlay a red line corresponding to their value
        struc_names = list(subject_data[struc].keys())
        metrics = [m for m in list(subject_data[struc][struc_names[0]].keys()) if m != 'slice_interp']
        nrows = len(struc_names) + 1
        ncols = len(metrics) + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = axes.flatten()
        idx = 0
        for i in range(ncols):
            if i == 0:
                axes[i].text(0.5, 0.5, "Structure name", fontsize=45, ha='center', va='center')
            else:
                # Load image 
                # img_path = os.path.join(ressources_path, f'imgs/{struc}_{metrics[i - 2]}.jpg')
                # axes[i].imshow(plt.imread(img_path))
                axes[i].text(0.5, 0.5, metrics[i-1], fontsize=45, ha='center', va='center')
            axes[i].set_axis_off()
            idx += 1
        for struc_name in struc_names:
            axes[idx].text(0.5, 0.5, struc_name, fontsize=45, ha='center', va='center')
            axes[idx].set_axis_off()
            idx += 1
            for metric in metrics:
                ax = axes[idx]
                y_subject = subject_data[struc][struc_name][metric]
                x_subject = subject_data[struc][struc_name]['slice_interp']

                # Keep lines with metrics line equal to metric
                all_values_data = all_values_df[struc][struc_name][metric]
                
                # Use seaborn line plot
                sns.lineplot(x='slice_interp', y='values', data=all_values_data, ax=ax, errorbar='sd')

                # Plot subject
                ax.plot(x_subject, y_subject, color='red', linewidth=2)
                
                # Add vertebrae labels
                top_pos = 0
                for disc, gap in discs_gap.items():
                    top_vert = rev_mapping[int(float(disc.split('-')[0]))].split('-')[0]
                    ax.axvline(x=top_pos, color='gray', linestyle='--', alpha=0.5)
                    ax.text(top_pos + gap//2, ax.get_ylim()[1], top_vert, verticalalignment='bottom', horizontalalignment='center', fontsize=12, color='black', alpha=0.7)
                    top_pos += gap
                # Add last vertebra
                last_vert = rev_mapping[int(float(disc.split('-')[0]))].split('-')[1]
                ax.axvline(x=top_pos, color='gray', linestyle='--', alpha=0.5)
                ax.text(top_pos + gap//2, ax.get_ylim()[1], last_vert, verticalalignment='bottom', horizontalalignment='center', fontsize=12, color='black', alpha=0.7)

                fig.tight_layout()
                idx += 1
            
        plt.savefig(str(ofolder_path / f"compared_{struc}.png"))
    
    # Create vertebrae, foramens figures
    for struc in ['foramens', 'vertebrae']:
        # Create a subplot for each subject and overlay a red line corresponding to their value
        struc_names = list(subject_data[struc].keys())
        metrics = list(subject_data[struc][struc_names[0]].keys())
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
                img = plt.imread(str(imgs_path / f'{img_name}.png'))
            else:
                img_name = f'{struc_name}'
                img_left = plt.imread(str(imgs_path / f'{img_name}_left.png'))
                img_right = plt.imread(str(imgs_path / f'{img_name}_right.png'))

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
                subject_value = subject_data[struc][struc_name][metric]
                all_values_data = all_values_df[struc][struc_name][metric]

                # Plot metric for subject
                # If subject_value >= mean_value, make the violin plot transparent
                if subject_value >= mean_dict[struc][struc_name][metric] or subject_value == -1:
                    sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='gray', alpha=0.2)
                else:
                    sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7)
                ax.tick_params(axis='x', rotation=45, labelsize=12)
                if subject_value != -1:
                    axes[idx].axvline(x=subject_value, color='red', linestyle='--')
                fig.tight_layout()
                idx += 1
            
        plt.savefig(str(ofolder_path / f"compared_{struc}.png"))
    
    # Create discs figures
    for struc in ['discs']:
        # Create a subplot for each subject and overlay a red line corresponding to their value
        struc_names = list(subject_data[struc].keys())
        metrics = list(subject_data[struc][struc_names[0]].keys())
        nrows = len(struc_names) + 1
        ncols = len(metrics) + 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = axes.flatten()
        idx = 0
        for i in range(ncols):
            if i == 0:
                axes[i].text(0.5, 0.5, "Structure name", fontsize=45, ha='center', va='center')
            elif i == 1:
                axes[i].text(0.5, 0.5, "Image", fontsize=45, ha='center', va='center')
            elif i == 2:
                axes[i].text(0.5, 0.5, "Segmentation", fontsize=45, ha='center', va='center')
            else:
                # Load image 
                img_path = os.path.join(ressources_path, f'imgs/{struc}_{metrics[i - 3]}.jpg')
                axes[i].imshow(plt.imread(img_path))
            axes[i].set_axis_off()
            idx += 1
        for struc_name in struc_names:
            axes[idx].text(0.5, 0.5, struc_name, fontsize=45, ha='center', va='center')
            axes[idx].set_axis_off()
            # Load images
            img_name = f'{struc}_{struc_name}'
            img = plt.imread(str(imgs_path / f'{img_name}_img.png'))
            seg = plt.imread(str(imgs_path / f'{img_name}_seg.png'))
            axes[idx+1].imshow(np.rot90(img), cmap='gray')
            axes[idx+1].set_axis_off()
            axes[idx+2].imshow(seg)
            axes[idx+2].set_axis_off()
            idx += 3
            for metric in metrics:
                ax = axes[idx]
                subject_value = subject_data[struc][struc_name][metric]
                if metric != 'intensity':
                    all_values_data = all_values_df[struc][struc_name][metric]
                    # Plot metric for subject
                    # If subject_value >= mean_value, make the violin plot transparent
                    if subject_value >= mean_dict[struc][struc_name][metric] or subject_value == -1:
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='gray', alpha=0.2)
                    else:
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7)
                    ax.tick_params(axis='x', rotation=45, labelsize=12)
                    if subject_value != -1:
                        axes[idx].axvline(x=subject_value, color='red', linestyle='--')
                else:
                    axes[idx].bar(range(len(subject_value)), subject_value)
                    axes[idx].set_xlabel("Slice index")
                    axes[idx].set_ylabel("Intensity")
                fig.tight_layout()
                idx += 1
            
        plt.savefig(str(ofolder_path / f"compared_{struc}.png"))


def convert_str_to_list(string):
    return [float(item.strip()) for item in string.split()[1:-1]]

if __name__ == "__main__":
    test_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/out/metrics_output'
    control_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/out/metrics_output'
    ofolder = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/code/totalspineseg/test'
    quiet = False
    generate_reports(
        test_path=test_path,
        control_path=control_path,
        ofolder_path=ofolder,
        quiet=quiet,
    )