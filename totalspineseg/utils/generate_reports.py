import importlib
import re
import os, argparse, textwrap
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import copy
from tqdm import tqdm
import totalspineseg.resources as ressources

def main():
    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes the outputs of totalspineseg/utils/measure_seg.py to generate a reports.
            It requires files to follow the BIDS naming conventions. 
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
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel proccess, defaults to multiprocessing.cpu_count().'
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
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            test_path = "{test_path}"
            control_path = "{control_path}"
            ofolder = "{ofolder}"
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    generate_reports(
        test_path=test_path,
        control_path=control_path,
        ofolder_path=ofolder,
        max_workers=max_workers,
        quiet=quiet
    )

def generate_reports(
        test_path,
        control_path,
        ofolder_path,
        max_workers,
        quiet
    ):
    # Load paths
    test_path = Path(test_path)
    control_path = Path(control_path)
    ofolder_path = Path(ofolder_path)

    # Load test demographics if exists
    if os.path.exists(str(test_path / "demographics.tsv")):
        df = pd.read_csv(str(test_path / "demographics.tsv"), sep='\t')
        demographics_test = {df.participant_id.iloc[i]:df_to_dict(df[df.participant_id == df.participant_id.iloc[i]]) for i in range(len(df))}
    else:
        demographics_test = {}
    
    # Load control demographics if exists
    if os.path.exists(str(control_path / "demographics.tsv")):
        df = pd.read_csv(str(control_path / "demographics.tsv"), sep='\t')
        demographics_control = {df.participant_id.iloc[i]:df_to_dict(df[df.participant_id == df.participant_id.iloc[i]]) for i in range(len(df))}
    else:
        demographics_control = {}
    
    # Load totalspineseg ressources path
    ressources_path = importlib.resources.files(ressources)
    with open(os.path.join(ressources_path, 'labels_maps/levels_maps.json'), 'r') as f:
        mapping = json.load(f)
        rev_mapping = {v: k for k, v in mapping.items()}
    
    # Extract metrics values of the control group
    if not os.path.exists(str(control_path / "all_values.json")):
        all_values = {'all':{}}
        subjects = [s for s in os.listdir(control_path) if os.path.isdir(control_path / s)]
        if not quiet: print("\n" "Processing control subjects:")
        for subject in tqdm(subjects, disable=quiet):
            control_sub_folder = control_path / subject
            sub_name = subject.split('_')[0]

            # Fetch demographics if available
            sex_key = None
            age_key = None
            if demographics_control and sub_name in demographics_control:
                if demographics_control[sub_name]["sex"] in ['M', 'F']:
                    sex_key = f'sex_{demographics_control[sub_name]["sex"]}'
                try:
                    float_age = float(demographics_control[sub_name]["age"])
                except ValueError:
                    float_age = None
                if float_age is not None:
                    age_key = f'age_{categorize_age_groups(float_age)}'                

            if sex_key is not None and not sex_key in all_values:
                all_values[sex_key] = {}
            if age_key is not None and not age_key in all_values:
                all_values[age_key] = {}

            # Compute metrics subject
            control_data = compute_metrics_subject(control_sub_folder)

            # Gather all values for each metric and structures
            for struc in control_data.keys():
                for struc_name in control_data[struc].keys():
                    for metric in control_data[struc][struc_name].keys():
                        # Add subject to all_values
                        subject_value = control_data[struc][struc_name][metric]
                        if subject_value != -1:
                            if struc not in all_values['all']:
                                all_values['all'][struc] = {}
                            if struc_name not in all_values['all'][struc]:
                                all_values['all'][struc][struc_name] = {}
                            if metric not in all_values['all'][struc][struc_name]:
                                all_values['all'][struc][struc_name][metric] = []
                            
                            if sex_key is not None:
                                if struc not in all_values[sex_key]:
                                    all_values[sex_key][struc] = {}
                                if struc_name not in all_values[sex_key][struc]:
                                    all_values[sex_key][struc][struc_name] = {}
                                if metric not in all_values[sex_key][struc][struc_name]:
                                    all_values[sex_key][struc][struc_name][metric] = []

                            if age_key is not None:
                                if struc not in all_values[age_key]:
                                    all_values[age_key][struc] = {}
                                if struc_name not in all_values[age_key][struc]:
                                    all_values[age_key][struc][struc_name] = {}
                                if metric not in all_values[age_key][struc][struc_name]:
                                    all_values[age_key][struc][struc_name][metric] = []

                            all_values['all'][struc][struc_name][metric].append(subject_value)
                            if sex_key is not None:
                                all_values[sex_key][struc][struc_name][metric].append(subject_value)
                            if age_key is not None:
                                all_values[age_key][struc][struc_name][metric].append(subject_value)

        # Align canal and CSF for control group
        all_values, discs_gap, last_disc = rescale_canal(all_values, rev_mapping)

        # TODO : save all values
    
    # Create global figures for test data subjects
    if not quiet: print("\n" "Generating test group reports:")
    create_figures_mp(test_path, ofolder_path, all_values, demographics_test, rev_mapping, discs_gap, last_disc, max_workers, quiet)

def df_to_dict(df):
    idx = df['participant_id'].keys()[0]
    d = {k:v[idx] for k,v in df.to_dict().items()}
    return d

def create_figures_mp(test_path, ofolder_path, all_values, demographics_test, rev_mapping, discs_gap, last_disc, max_workers, quiet):
    # Create a list of test subject folders
    test_sub_folders = [test_path / subject for subject in os.listdir(test_path)]
    imgs_paths = [test_sub_folder / 'imgs' for test_sub_folder in test_sub_folders]
    ofolder_subjects = [ofolder_path / subject for subject in os.listdir(test_path)]

    process_map(
        partial(
            create_figures,
            all_values=all_values,
            demographics_test=demographics_test,
            rev_mapping=rev_mapping,
            discs_gap=discs_gap,
            last_disc=last_disc
        ),
        test_sub_folders,
        imgs_paths,
        ofolder_subjects,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )
    #create_figures(test_sub_folders[0], imgs_paths[0], ofolder_subjects[0], all_values, demographics_test, rev_mapping, discs_gap, last_disc)

def create_figures(sub_folder, imgs_path, ofolder_subject, all_values, demographics_test, rev_mapping, discs_gap, last_disc):
    # Load subject data
    subject_data = compute_metrics_subject(sub_folder)
    sub_name = sub_folder.name.split('_')[0]

    # Filtrate all_values if demographics are available
    sex_group = None
    age_group = None
    new_all_values = {'all': all_values['all']}
    if demographics_test and sub_name in demographics_test:
        age = demographics_test[sub_name]['age']
        sex = demographics_test[sub_name]['sex']
        if sex in ['M', 'F']:
            sex_group = f'sex_{sex}'
        try:
            float_age = float(age)
        except ValueError:
            float_age = None
        if float_age is not None:
            age_group = f'age_{categorize_age_groups(float_age)}'  
    
    if sex_group is not None and sex_group in all_values.keys():
        new_all_values[sex_group] = all_values[sex_group]
    if age_group is not None and age_group in all_values.keys():
        new_all_values[age_group] = all_values[age_group]
    all_values = copy.deepcopy(new_all_values)

    # Create median and std dictionary
    median_dict = {}
    new_all_values = copy.deepcopy(all_values)
    for group in all_values.keys():
        for struc in all_values[group].keys():
            if struc in ['foramens', 'discs', 'vertebrae']:
                for struc_name in all_values[group][struc].keys():
                    for metric, values in all_values[group][struc][struc_name].items():
                        # Discard values at 4 times the std from the median
                        median_value = np.median(values)
                        std_value = np.std(values)
                        new_values = [v for v in values if v >= median_value - 4*std_value and v <= median_value + 4*std_value and v != -1]
                        new_all_values[group][struc][struc_name][metric] = new_values
                        median_value = np.median(new_values)
                        std_value = np.std(new_values)
                        if group not in median_dict:
                            median_dict[group] = {}
                        if struc not in median_dict[group]:
                            median_dict[group][struc] = {struc_name: {metric: {'median': median_value, 'std': std_value}}}
                        if struc_name not in median_dict[group][struc]:
                            median_dict[group][struc][struc_name] = {metric: {'median': median_value, 'std': std_value}}
                        if metric not in median_dict[group][struc][struc_name]:
                            median_dict[group][struc][struc_name][metric] = {'median': median_value, 'std': std_value}
    
    # Compute discs gradings
    subject_data = compute_discs_gradings(subject_data, new_all_values)
    
    # Convert all_values to dataframe
    all_values_df = convert_to_df(new_all_values)

    # Rescale canal and CSF metrics
    interp_data = copy.deepcopy(subject_data)
    for struc in ['canal', 'csf']:
        for struc_name in subject_data[struc].keys():
            for metric in subject_data[struc][struc_name].keys():
                if metric in ['slice_nb', 'disc_level']:
                    continue
                interp_values, slice_interp = rescale_with_discs(subject_data[struc][struc_name]['disc_level'], subject_data[struc][struc_name][metric], rev_mapping, discs_gap, last_disc)
                interp_data[struc][struc_name][metric] = interp_values
            interp_data[struc][struc_name]['slice_interp'] = slice_interp
            # remove slice_nb and disc_level from dict
            interp_data[struc][struc_name].pop('slice_nb', None)
            interp_data[struc][struc_name].pop('disc_level', None)

    # Create figures    
    ofolder_subject.mkdir(parents=True, exist_ok=True)
    create_global_figures(interp_data, all_values_df, discs_gap, last_disc, median_dict, imgs_path, rev_mapping, ofolder_subject)

def compute_discs_gradings(subject_data, all_values):
    for group in all_values.keys():
        for disc in subject_data['discs'].keys():
            # Grade disc based on height compared to median height in all_values
            if 'grading' not in subject_data['discs'][disc]:
                subject_data['discs'][disc]['grading'] = {}
            median_height = None
            std_height = None
            if disc in all_values[group]['discs']:
                if 'median_thickness' in all_values[group]['discs'][disc]:
                    median_height = np.median(all_values[group]['discs'][disc]['median_thickness'])
                    std_height = np.std(all_values[group]['discs'][disc]['median_thickness'])
            if median_height is not None and std_height is not None:
                disc_height = subject_data['discs'][disc]['median_thickness']
                disc_intensity = subject_data['discs'][disc]['intensity_variation']
                if disc_height <= 0.3*median_height:
                    grade = 8
                elif disc_height <= 0.6*median_height:
                    grade = 7
                elif disc_height <= 0.9*median_height:
                    grade = 6
                elif disc_intensity <= 0.2:
                    grade = 5
                elif disc_intensity <= 0.4:
                    grade = 4
                elif disc_intensity <= 0.55:
                    grade = 3
                elif disc_intensity <= 0.7:
                    grade = 2
                elif disc_intensity > 0.7:
                    grade = 1
                subject_data['discs'][disc]['grading'][group] = grade
            else:
                subject_data['discs'][disc]['grading'][group] = 'Error'
    return subject_data

def convert_to_df(all_values):
    new_values = copy.deepcopy(all_values)
    for group in new_values.keys():
        for struc in new_values[group].keys():
            for struc_name in new_values[group][struc].keys():
                # Convert dict to dataframe with keys as columns and lines as subjects
                # Prepare a dictionary where each key is a metric and each value is a list of values for all subjects
                for i, metric in enumerate(all_values[group][struc][struc_name].keys()):
                    if metric not in ['discs_gap', 'slice_interp']:
                        data = {'subjects' : [], 'values' : [], 'slice_interp' : []}
                        for j, subject_value in enumerate(all_values[group][struc][struc_name][metric]):
                            if isinstance(subject_value, list):
                                for value, slice_interp in zip(subject_value, all_values[group][struc][struc_name]['slice_interp'][j]):
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
                    new_values[group][struc][struc_name][metric] = df
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
    
    # Compute discs metrics
    merged_data = compute_discs_metrics(merged_data)

    # Compute foramen metrics
    merged_data = compute_foramens_metrics(merged_data)

    # Compute vertebrae metrics
    merged_data = compute_vertebrae_metrics(merged_data)
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
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def process_foramens(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def compute_discs_metrics(data_dict):
    # Compute Disc Height Index (DHI)
    for struc_name in data_dict['discs'].keys():
        top_vertebra = struc_name.split('-')[0]
        if top_vertebra in data_dict['vertebrae']:
            # Normalize disc height with top vertebra AP_thickness
            data_dict['discs'][struc_name]['DHI'] = data_dict['discs'][struc_name]['median_thickness'] / data_dict['vertebrae'][top_vertebra]['AP_thickness']
        else:
            data_dict['discs'][struc_name]['DHI'] = -1
    return data_dict

def compute_foramens_metrics(data_dict):
    # Compute Foramen metrics
    for struc_name in data_dict['foramens'].keys():
        top_vertebra = struc_name.replace('foramens_','').split('-')[0]
        if not top_vertebra in data_dict['vertebrae']:
            data_dict['foramens'][struc_name]['right_surface'] = -1
            data_dict['foramens'][struc_name]['left_surface'] = -1
            data_dict['foramens'][struc_name]['asymmetry_R-L'] = -1
        else:
            # Normalize foramen surfaces with top vertebra AP thickness
            for surface in ['right_surface', 'left_surface']:
                if data_dict['foramens'][struc_name][surface] != -1:
                    data_dict['foramens'][struc_name][surface] = data_dict['foramens'][struc_name][surface] / (data_dict['vertebrae'][top_vertebra]['AP_thickness']*data_dict['vertebrae'][top_vertebra]['median_thickness'])

            # Create asymmetry quotient
            if data_dict['foramens'][struc_name]['right_surface'] != -1 and data_dict['foramens'][struc_name]['left_surface'] != -1 and data_dict['foramens'][struc_name]['left_surface'] != 0:
                data_dict['foramens'][struc_name]['asymmetry_R-L'] = data_dict['foramens'][struc_name]['right_surface'] / data_dict['foramens'][struc_name]['left_surface']
            else:
                data_dict['foramens'][struc_name]['asymmetry_R-L'] = -1
    return data_dict

def compute_vertebrae_metrics(data_dict):
    # Compute Vertebrae metrics
    return data_dict

def rescale_canal(all_values, rev_mapping):
    '''
    Rescale subject canals and CSF based on discs z coordinates.
    '''
    new_values = copy.deepcopy(all_values)
    struc = 'canal'
    struc_name = 'canal'
    # Align all metrics for each subject using discs level as references
    disc_levels = all_values['all'][struc][struc_name]['disc_level']
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
    gap_list = []
    discs_list = []
    for k, v in gap_dict.items():
        gap_list.append(int(round(np.median(v))))
        discs_list.append(k.split('-')[0])
        discs_list.append(k.split('-')[1])
    discs_list = [int(float(v)) for v in list(np.unique(discs_list))]
    discs_gap = int(round(np.median(gap_list)))
    last_disc = rev_mapping[max(discs_list)]

    for key in all_values.keys():
        for struc in ['canal', 'csf']:
            for struc_name in all_values[key][struc].keys():
                # Rescale subjects
                add_slice_interp = True
                for metric in all_values[key][struc][struc_name].keys():
                    if metric in ['slice_nb', 'disc_level']:
                        continue
                    for subj_idx in range(len(all_values[key][struc][struc_name][metric])):
                        interp_values, slice_interp = rescale_with_discs(all_values[key][struc][struc_name]['disc_level'][subj_idx], all_values[key][struc][struc_name][metric][subj_idx], rev_mapping, discs_gap, last_disc)
                        new_values[key][struc][struc_name][metric][subj_idx] = interp_values
                        if 'slice_interp' not in new_values[key][struc][struc_name]:
                            new_values[key][struc][struc_name]['slice_interp'] = []
                        if add_slice_interp:
                            new_values[key][struc][struc_name]['slice_interp'].append(slice_interp)
                    add_slice_interp = False
                # Remove slice_nb and disc_level from dict
                new_values[key][struc][struc_name].pop('slice_nb', None)
                new_values[key][struc][struc_name].pop('disc_level', None)

    return new_values, discs_gap, last_disc

def rescale_with_discs(disc_levels, metric_list, rev_mapping, gap, last_disc):
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
            yp = values[subj_disc_positions[disc_idx]:subj_disc_positions[disc_idx+1]]
            xp = np.linspace(0, gap-1, len(yp))
            x = np.linspace(0, gap-1, gap)
            if not -1 in yp:
                if yp.size > 0:
                    interp_func = np.interp(
                        x=x,
                        xp=xp,
                        fp=yp
                    )
                else:
                    interp_func = np.full_like(x, 0)
            else:
                interp_func = np.full_like(x, -1)
            interp_values += interp_func.tolist()

    start_disc_gap = 0
    disc = last_disc
    while disc != rev_mapping[int(subj_disc_values[0])]:
        start_disc_gap += gap
        disc = previous_structure(disc)
    slice_interp += list(range(start_disc_gap, start_disc_gap + len(interp_values)))
    return interp_values, slice_interp

def previous_structure(structure_name):
    '''
    Return the name of the previous structure in anatomical order.
    '''
    structure_name = structure_name.strip()

    # Handle discs (L5-S1, L4-L5, ..., T9-T10)
    # and foramens (foramens_L5-S1, foramens_L4-L5, ..., foramens_T9-T10)
    foramen = False
    if '-' in structure_name:
        if structure_name.startswith('foramens_'):
            foramen = True
            structure_name = structure_name.replace('foramens_', '')
        parts = structure_name.split('-')
        if len(parts) == 2:
            # Get previous vertebra
            lower = []
            for part in parts:
                next_lower = previous_vertebra(part)
                lower.append(next_lower)
            previous_structure = "-".join(lower)
            if foramen:
                previous_structure = "foramens_" + previous_structure
            return previous_structure
        else:
            return None
        
    # Handle vertebrae (T9, T10, T11, T12, L1, L2, L3, L4, L5, S1)
    elif re.match(r'^[TLCS]\d+$', structure_name):
        return previous_vertebra(structure_name)

def previous_vertebra(vertebra):
    vertebra = vertebra.strip()
    if re.match(r'^T\d+$', vertebra):
        if int(vertebra[1:]) == 1:
            next_lower = "C7"
        else:
            next_lower = f"T{int(vertebra[1:]) - 1}"
    elif re.match(r'^L\d+$', vertebra):
        if int(vertebra[1:]) == 1:
            next_lower = "T12"
        else:
            next_lower = f"L{int(vertebra[1:]) - 1}"
    elif re.match(r'^S', vertebra):
        next_lower = "L5"
    elif re.match(r'^C\d+$', vertebra):
        if int(vertebra[1:]) == 1:
            return None
        else:
            next_lower = f"C{int(vertebra[1:]) - 1}"
    return next_lower

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
            if column != 'center':
                struc_dict[column] = struc_data[column][struc_idx]
        subject_dict[struc] = struc_dict
    return subject_dict

def create_global_figures(subject_data, all_values_df, discs_gap, last_disc, median_dict, imgs_path, rev_mapping, ofolder_path):
    """
    Create global figures from the processed subjects data.

    Parameters:
        subject_data (dict): A dictionary containing the subject data.
        all_values (dict): A dictionary containing all processed metrics data for each control subject.
        imgs_path (Path): Path to the folder containing the subject images.
        ofolder_path (Path): Path to the output folder where reports will be saved.
    """
    ressources_path = importlib.resources.files(ressources)

    metrics_dict = {
            'discs': ['median_thickness', 'DHI', 'volume', 'eccentricity', 'solidity'],
            'vertebrae': ['median_thickness', 'AP_thickness', 'volume'],
            'foramens': ['right_surface', 'left_surface', 'asymmetry_R-L'],
            'canal': ['area', 'diameter_AP', 'diameter_RL', 'eccentricity', 'solidity'],
        }

    # Create discs, vertebrae, foramens figures
    for group in all_values_df.keys():
        for struc in ['canal']:
            # Create a subplot for each subject and overlay a red line corresponding to their value
            struc_names = np.array(list(subject_data[struc].keys()))
            struc_names = struc_names[np.isin(struc_names, list(all_values_df[group][struc].keys()))].tolist()
            metrics = metrics_dict[struc]
            nrows = len(struc_names) + 1
            ncols = len(metrics) + 1
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
            axes = axes.flatten()
            idx = 0
            for i in range(ncols):
                if i == 0:
                    axes[i].text(0.5, 0.5, "Structure name", fontsize=45, ha='center', va='center', fontweight='bold')
                else:
                    if os.path.exists(os.path.join(ressources_path, f'imgs/{struc}_{metrics[i - 1]}.jpg')):
                        # Load image 
                        img_path = os.path.join(ressources_path, f'imgs/{struc}_{metrics[i - 1]}.jpg')
                        axes[i].imshow(plt.imread(img_path))
                    else:
                        axes[i].text(0.5, 0.5, metrics[i-1], fontsize=45, ha='center', va='center', fontweight='bold')
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
                    all_values_data = all_values_df[group][struc][struc_name][metric]
                    
                    # Use seaborn line plot
                    sns.lineplot(x='slice_interp', y='values', data=all_values_data, ax=ax, errorbar='sd')

                    # Plot subject
                    ax.plot(x_subject, y_subject, color='red', linewidth=2)
                    
                    # Add vertebrae labels
                    disc = last_disc
                    top_pos = 0
                    nb_discs = all_values_data['slice_interp'].max()//discs_gap
                    for i in range(nb_discs+1):
                        top_vert = disc.split('-')[0]
                        ax.axvline(x=top_pos, color='gray', linestyle='--', alpha=0.5)
                        ax.text(top_pos + discs_gap//2, ax.get_ylim()[1], top_vert, verticalalignment='bottom', horizontalalignment='center', fontsize=12, color='black', alpha=0.7)
                        top_pos += discs_gap
                        disc = previous_structure(disc)

                    ax.set_xlabel('')
                    fig.tight_layout()
                    idx += 1

            plt.savefig(str(ofolder_path / f"compared_{group}_{struc}.png"))

        # Create vertebrae, foramens figures
        for struc in ['foramens']:
            # Create a subplot for each subject and overlay a red line corresponding to their value
            struc_names = np.array(list(subject_data[struc].keys()))
            struc_names = struc_names[np.isin(struc_names, list(all_values_df[group][struc].keys()))].tolist()
            metrics = metrics_dict[struc]
            nrows = len(struc_names) + 1
            ncols = len(metrics) + 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
            axes = axes.flatten()
            idx = 0
            for i in range(ncols):
                if i == 0:
                    axes[i].text(0.5, 0.5, "Structure name", fontsize=45, ha='center', va='center', fontweight='bold')
                elif i == 1:
                    axes[i].text(0.5, 0.5, "Segmentation", fontsize=45, ha='center', va='center', fontweight='bold')
                else:
                    if os.path.exists(os.path.join(ressources_path, f'imgs/{struc}_{metrics[i - 2]}.jpg')):
                        # Load image 
                        img_path = os.path.join(ressources_path, f'imgs/{struc}_{metrics[i - 2]}.jpg')
                        axes[i].imshow(plt.imread(img_path))
                    else:
                        axes[i].text(0.5, 0.5, metrics[i - 2], fontsize=45, ha='center', va='center', fontweight='bold')
                
                axes[i].set_axis_off()
                idx += 1
            for struc_name in struc_names:
                axes[idx].text(0.5, 0.5, struc_name, fontsize=45, ha='center', va='center')
                axes[idx].set_axis_off()
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
                    all_values_data = all_values_df[group][struc][struc_name][metric]

                    # Plot metric for subject
                    if subject_value == -1:
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='gray', alpha=0.2)
                    elif subject_value < median_dict[group][struc][struc_name][metric]['median'] - 0.8*median_dict[group][struc][struc_name][metric]['std']:
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='orange')
                    elif subject_value > median_dict[group][struc][struc_name][metric]['median'] + 0.8*median_dict[group][struc][struc_name][metric]['std']:
                        # Highlight the violin plot
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='green')
                    else:
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='gray', alpha=0.2)
                        
                    ax.tick_params(axis='x', rotation=45, labelsize=12)
                    if subject_value != -1:
                        axes[idx].axvline(x=subject_value, color='red', linestyle='--')
                    ax.set_xlabel('')
                    fig.tight_layout()
                    idx += 1

            plt.savefig(str(ofolder_path / f"compared_{group}_{struc}.png"))

        # Create discs figures
        for struc in ['discs']:
            # Create a subplot for each subject and overlay a red line corresponding to their value
            struc_names = np.array(list(subject_data[struc].keys()))
            struc_names = struc_names[np.isin(struc_names, list(all_values_df[group][struc].keys()))].tolist()
            metrics = metrics_dict[struc]
            nrows = len(struc_names) + 1
            ncols = len(metrics) + 4
            fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 7 * nrows))
            fig.subplots_adjust(bottom=0)
            axes = axes.flatten()
            idx = 0
            for i in range(ncols):
                if i == 0:
                    axes[i].text(0.5, 0.5, "Structure name", fontsize=70, ha='center', va='center', fontweight='bold')
                elif i == 1:
                    axes[i].text(0.5, 0.5, "Disc grading", fontsize=70, ha='center', va='center', fontweight='bold')
                elif i == 2:
                    axes[i].text(0.5, 0.5, "Image", fontsize=70, ha='center', va='center', fontweight='bold')
                elif i == 3:
                    axes[i].text(0.5, 0.5, "Segmentation", fontsize=70, ha='center', va='center', fontweight='bold')
                else:
                    if os.path.exists(os.path.join(ressources_path, f'imgs/{struc}_{metrics[i - 4]}.jpg')):
                        # Load image 
                        img_path = os.path.join(ressources_path, f'imgs/{struc}_{metrics[i - 4]}.jpg')
                        axes[i].imshow(plt.imread(img_path))
                    else:
                        axes[i].text(0.5, 0.5, metrics[i - 4], fontsize=70, ha='center', va='center', fontweight='bold')
                axes[i].set_axis_off()
                idx += 1
            for struc_name in struc_names:
                axes[idx].text(0.5, 0.5, struc_name, fontsize=70, ha='center', va='center')
                axes[idx].set_axis_off()
                grading = subject_data[struc][struc_name]['grading'][group]
                axes[idx+1].text(0.5, 0.5, f'Grading {grading}', fontsize=70, ha='center', va='center')
                axes[idx+1].set_axis_off()
                # Load images
                img_name = f'{struc}_{struc_name}'
                img = plt.imread(str(imgs_path / f'{img_name}_img.png'))
                seg = plt.imread(str(imgs_path / f'{img_name}_seg.png'))
                axes[idx+2].imshow(np.rot90(img), cmap='gray')
                axes[idx+2].set_axis_off()
                axes[idx+3].imshow(np.rot90(seg))
                axes[idx+3].set_axis_off()
                idx += 4
                for metric in metrics:
                    ax = axes[idx]
                    subject_value = subject_data[struc][struc_name][metric]
                    all_values_data = all_values_df[group][struc][struc_name][metric]
                    # Plot metric for subject
                    if subject_value == -1:
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='gray', alpha=0.2)
                    elif subject_value < median_dict[group][struc][struc_name][metric]['median'] - 0.8*median_dict[group][struc][struc_name][metric]['std']:
                        # Highlight the violin plot in orange
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='orange')
                    elif subject_value > median_dict[group][struc][struc_name][metric]['median'] + 0.8*median_dict[group][struc][struc_name][metric]['std']:
                        # Highlight the violin plot in green
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='green')
                    else:
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='gray', alpha=0.2)

                    ax.tick_params(axis='x', rotation=45, labelsize=30)
                    if subject_value != -1:
                        axes[idx].axvline(x=subject_value, color='red', linestyle='--')
                    ax.set_xlabel('')
                    fig.tight_layout()
                    idx += 1

            plt.savefig(str(ofolder_path / f"compared_{group}_{struc}.png"))
        
        for struc in ['vertebrae']:
            # Create a subplot for each subject and overlay a red line corresponding to their value
            struc_names = np.array(list(subject_data[struc].keys()))
            struc_names = struc_names[np.isin(struc_names, list(all_values_df[group][struc].keys()))].tolist()
            metrics = metrics_dict[struc]
            nrows = len(struc_names) + 1
            ncols = len(metrics) + 3
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
            axes = axes.flatten()
            idx = 0
            for i in range(ncols):
                if i == 0:
                    axes[i].text(0.5, 0.5, "Structure name", fontsize=45, ha='center', va='center', fontweight='bold')
                elif i == 1:
                    axes[i].text(0.5, 0.5, "Image", fontsize=45, ha='center', va='center', fontweight='bold')
                elif i == 2:
                    axes[i].text(0.5, 0.5, "Segmentation", fontsize=45, ha='center', va='center', fontweight='bold')
                else:
                    if os.path.exists(os.path.join(ressources_path, f'imgs/{struc}_{metrics[i - 3]}.jpg')):
                        # Load image 
                        img_path = os.path.join(ressources_path, f'imgs/{struc}_{metrics[i - 3]}.jpg')
                        axes[i].imshow(plt.imread(img_path))
                    else:
                        axes[i].text(0.5, 0.5, metrics[i - 3], fontsize=45, ha='center', va='center', fontweight='bold')
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
                axes[idx+2].imshow(np.rot90(seg))
                axes[idx+2].set_axis_off()
                idx += 3
                for metric in metrics:
                    ax = axes[idx]
                    subject_value = subject_data[struc][struc_name][metric]
                    all_values_data = all_values_df[group][struc][struc_name][metric]
                    # Plot metric for subject
                    if subject_value == -1:
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='gray', alpha=0.2)
                    elif subject_value < median_dict[group][struc][struc_name][metric]['median'] - 0.8*median_dict[group][struc][struc_name][metric]['std']:
                        # Highlight the violin plot in orange
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='orange')
                    elif subject_value > median_dict[group][struc][struc_name][metric]['median'] + 0.8*median_dict[group][struc][struc_name][metric]['std']:
                        # Highlight the violin plot in green
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='green')
                    else:
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='gray', alpha=0.2)

                    ax.tick_params(axis='x', rotation=45, labelsize=12)
                    if subject_value != -1:
                        axes[idx].axvline(x=subject_value, color='red', linestyle='--')
                    ax.set_xlabel('')
                    fig.tight_layout()
                    idx += 1

            plt.savefig(str(ofolder_path / f"compared_{group}_{struc}.png"))


def convert_str_to_list(string):
    return [float(item.strip()) for item in string[1:-1].split(',')]


def categorize_age_groups(age):
    """
    Categorize age into groups.
    
    Args:
        age: Age value or list of ages
    
    Returns:
        Age group label
    """
    if age < 40:
        return '18-39'
    elif age < 60:
        return '40-59'
    else:
        return '60+'

if __name__ == "__main__":
    test_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/metrics_output'
    control_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/metrics_output'
    ofolder = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/code/totalspineseg/test'
    quiet = False
    generate_reports(
        test_path=test_path,
        control_path=control_path,
        ofolder_path=ofolder,
        max_workers=8,
        quiet=quiet,
    )