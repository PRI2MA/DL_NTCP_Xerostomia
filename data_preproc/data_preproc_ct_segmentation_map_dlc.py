"""
This script:
    - main(): Create DLC segmentation_map, and for the output segmentation_map (segmentation_map_dlc.npy):
        - Start with zeros Numpy array (with same shape as DLC segmentation_map).
        - For each relevant structure:
            - Add its voxels to initial zeros array using own consistent value map.
            - Save the segmentation (segmentation_map_dlc.npy) in 'segmentation_map_dlc' folder.

IMPORTANT NOTES:
    - The segmentation_map (segmentation_map_dlc.npy) will only contain the relevant structures and with values from our
        own consistent value map.
    - For some patients + RTSTRUCT + structures: there are contouring issues, e.g. when there are no or too less
        contour coordinates (just 1 coordinate? See try-except operation in get_segmentation_map()). The structures for
        which this hold can be found in the files 'structures_with_contouring_issues_{i}.txt of the corresponding
        patient.

ASSUMPTION: every patient has one DLC RTSTRUCT file. (Note: normally patients can have multiple RTSTRUCT folders and/or
    multiple RTSTRUCT files.)

The read_ds() and get_segmentation_map() functions are from http://aapmchallenges.cloudapp.net/forums/3/2/.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import pydicom as pdcm
from tqdm import tqdm
from glob import glob
from pydicom import dcmread

import data_preproc_config as cfg
from data_preproc_functions import (create_folder_if_not_exists, get_all_folders, list_to_txt_str, Logger, set_default,
                                    sort_human)
from data_preproc_ct_segmentation_map_citor import (read_ds, get_segmentation_map, main_overview_segmentation_map)


def modify_structure_dlc(structure, logger):
    """
    This script applies modifications to (try to) get consistent notations of the structure as possible,
    across different files.

    Args:
        structure (str): structure, e.g. 'DLC_Parotid_L'
        logger:
    Returns:

    """
    # Create dictionary for mapping structure. 'key' will be replaced by 'value' in 'structure'
    source_replacement_dict = {
        'DLC_Parotid_L': 'parotis_li',
        'DLC_Parotid_R': 'parotis_re',
        'DLC_Submandibular_L': 'submandibularis_li',
        'DLC_Submandibular_R': 'submandibularis_re',
        'DLC_Crico': 'crico',
        'DLC_Esophagus_Cerv': 'esophagus_cerv',
        'DLC_Thyroid': 'thyroid',
        'DLC_Buccalmucosa_L': 'buccalmucosa_li',
        'DLC_Buccalmucosa_R': 'buccalmucosa_re',
        'DLC_Glotticarea': 'glotticarea',
        'DLC_Mandible': 'mandible',
        'DLC_OralCavity_Ext': 'oralcavity_ext',
        # 'DLC_PCM': 'pcm',
        'DLC_PCM_Inf': 'pcm_inf',
        'DLC_PCM_Med': 'pcm_med',
        'DLC_PCM_Sup': 'pcm_sup',
        'DLC_Supraglottic': 'supraglottic'
    }

    # E.g. if structure = 'foo_DLC_Parotid_L_bar', then replace it by the consistent structure name 'parotid_li'
    # (because 'DLC_Parotid_L'.lower() in structure.lower())
    consistent_structure_names = [v for k, v in source_replacement_dict.items() if k.lower() == structure.lower()]
    # If len(structures) == 0, then only apply lowercase to 'structure'
    if len(set(consistent_structure_names)) == 1:
        structure = consistent_structure_names[0]
    elif len(consistent_structure_names) > 1:
        logger.my_print('structure: {}'.format(structure))
        logger.my_print('consistent_structure_names: {}'.format(consistent_structure_names))
        assert False

    return structure.lower()


def main_segmentation_map():
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    # data_dir_ct = cfg.data_dir
    data_dir_dlc = cfg.data_dir_dlc
    save_root_dir = cfg.save_root_dir
    save_dir_segmentation_map_dlc = cfg.save_dir_segmentation_map_dlc
    patient_folder_types = cfg.patient_folder_types
    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_ct_segmentation_map_dlc_logging_txt))
    patients_with_contouring_issues = list()
    start = time.time()

    # Create folder if not exist
    create_folder_if_not_exists(save_dir_segmentation_map_dlc)

    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir_ct = cfg.data_dir.format(d_i)

        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # Get all patient_ids
        logger.my_print('Listing all patient ids and their files...')
        patients_dlc_list = os.listdir(data_dir_dlc)
        patients_dlc_list = sort_human(patients_dlc_list)
        n_patients = len(patients_dlc_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_dlc_list = [x for x in test_patients_list if x in patients_dlc_list]

        for patient_id in tqdm(patients_dlc_list):
            logger.my_print('Patient_id: {id}'.format(id=patient_id))

            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            # Get all folders and sub-folders
            all_folders_ct = get_all_folders(os.path.join(data_dir_ct, patient_id, folder_type_i))
            all_folders_dlc = get_all_folders(os.path.join(data_dir_dlc, patient_id, folder_type_i))

            # Get CT folder
            # Assumption: patients only have one CT folder
            path_ct = [x for x in all_folders_ct if x.endswith('\\CT')]

            # 'with_contrast' has highest priority: if it does not contain one or more of the required folders,
            # then search in 'no_contrast'
            # TODO: temporary, for MDACC
            if use_umcg:
                if len(path_ct) == 0:
                    all_folders_ct_2 = get_all_folders(os.path.join(data_dir_ct, patient_id, patient_folder_types[1]))
                    path_ct = [x for x in all_folders_ct_2 if x.endswith('\\CT')]

            assert len(path_ct) == 1
            path_ct = path_ct[0]

            # Load CT slices
            slices = [pdcm.read_file(dcm) for dcm in glob(path_ct + '/*')]
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            logger.my_print('Number of CT slices: {n}'.format(n=len(slices)))
            image = np.stack([s.pixel_array for s in slices], axis=-1)

            # Get RTSTRUCT file
            if use_umcg:
                path_rtss = [x for x in all_folders_dlc if x.endswith('\\RTSTRUCT')]

                # 'with_contrast' has highest priority: if it does not contain one or more of the required folders,
                # then search in 'no_contrast'
                if len(path_rtss) == 0:
                    all_folders_dlc_2 = get_all_folders(os.path.join(data_dir_dlc, patient_id, patient_folder_types[1]))
                    path_rtss = [x for x in all_folders_dlc_2 if x.endswith('\\RTSTRUCT')]

                assert len(path_rtss) == 1
                path_rtss = path_rtss[0]
                file_rtss = glob(path_rtss + '/*')
                assert len(file_rtss) == 1
                file_rtss = file_rtss[0]
            else:
                # TODO: temporary, for MDACC
                path_rtss = [x for x in all_folders_dlc if x.endswith('\\Post processed RTSS')]
                assert len(path_rtss) == 1
                path_rtss = path_rtss[0]
                file_rtss = glob(path_rtss + '/*')
                assert len(file_rtss) == 1
                file_rtss = file_rtss[0]

            # Load RTSTRUCT
            ds = dcmread(file_rtss)

            # Extract contour information from structures
            contours = read_ds(ds)

            # Extract segmentation_map and potentially a list of structures for which there are issues with
            # extracting contours
            segmentation_map, colors, structure_con_issues_i = get_segmentation_map(
                contours=contours, slices=slices, image=image, modify_structure=modify_structure_dlc, logger=logger)

            # Create structures_value_count.json
            # Available structures, e.g. 'parotid_re', 'parotid_li', 'myelum', etc.
            structures = dict()
            structures[0] = 'background'
            for item in ds.StructureSetROISequence:
                structures[item.ROINumber] = item.ROIName

            assert len(structures) >= len(contours)

            # Example of 'unique': array([ 0,  6,  7,  8,  9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30,
            # 31], dtype=uint8)
            # Example of 'counts': array([30406772, 15522, 18330, 17897, 5104, 5536, 391993, 290169, 4942,
            # 2981, 2600, 2715, 6809, 156, 1452, 24, 19533, 1437, 1164], dtype=int64)
            unique, counts = np.unique(segmentation_map.flatten(), return_counts=True)

            structures_value_count = dict()
            for value, structure in structures.items():
                # Map variable data type to make sure that the data types are consistent
                value = int(value)
                structure = str(structure)

                # Check if structure_value (0, 1, ..., n) is in segmentation_map, because segmentation_map may not contain
                # certain structures!
                structure_value_in_segmentation_map = np.any(unique == value)

                if structure_value_in_segmentation_map:
                    idx_of_array = np.where(unique == value)  # tuple, e.g.: (array([0], dtype=int64),)
                    assert len(idx_of_array) == 1  # check
                    idx_of_array = idx_of_array[0]  # np.array(), e.g.: array([0], dtype=int64)
                    assert len(idx_of_array) == 1  # check
                    idx_of_array = idx_of_array[0]  # int, e.g.: 0
                    count = counts[idx_of_array]
                else:
                    count = 0

                # Construct value-count dictionary for the given structure
                count = int(count)
                structure_dict = dict()
                structure_dict['Value'] = value
                structure_dict['Count'] = count
                structure_dict['Modified_name'] = modify_structure_dlc(structure=structure, logger=logger)
                structures_value_count[structure] = structure_dict

            # For each relevant structure: store its segmentation_map in output_segmentation_map with value from own
            # value map
            output_segmentation_map = np.zeros_like(segmentation_map)
            output_segmentation_map_dtype = output_segmentation_map.dtype
            for structure in cfg.structures_uncompound_list:
                # Get original segmentation value
                structure_value_list = [v['Value'] for _, v in structures_value_count.items() if
                                        v['Modified_name'] == structure]

                # Add structure's voxels to output_segmentation_map array with value from own value map
                # Note that the structure may not be present in any RTSTRUCT and hence len(structure_value_list) == 0
                assert len(structure_value_list) <= 1
                new_structure_value = cfg.structures_values[structure]
                if len(structure_value_list) == 1:
                    original_structure_value = structure_value_list[0]
                    output_segmentation_map += ((segmentation_map == original_structure_value) * new_structure_value) \
                        .astype(output_segmentation_map_dtype)
                else:
                    logger.my_print('Patient has no segmentation of {structure} in any RTSTRUCT file'
                                    .format(structure=structure), level='warning')

            # Save results
            save_path = os.path.join(save_dir_segmentation_map_dlc, patient_id)
            create_folder_if_not_exists(save_path)

            # Save structures_value_count
            with open(os.path.join(save_path, cfg.filename_structures_value_count_json), 'w') as f:
                json.dump(structures_value_count, f, default=set_default)

            # Save output_segmentation_map as Numpy array
            logger.my_print('output_segmentation_map.shape: {}'.format(output_segmentation_map.shape))
            np.save(file=os.path.join(save_path, cfg.filename_segmentation_map_npy), arr=output_segmentation_map)

            # Save structures for which there are contouring issues
            if len(structure_con_issues_i) > 0:
                patients_with_contouring_issues.append(patient_id)
                with open(os.path.join(save_path, cfg.filename_structures_with_contouring_issues_dlc_txt), 'w') as f:
                    f.write(list_to_txt_str(structure_con_issues_i))

        # Save list of patient_ids for which there is at least one structure with contouring issues
        with open(os.path.join(save_root_dir, cfg.filename_patients_with_contouring_issues_dlc_txt), 'w') as f:
            f.write(list_to_txt_str(patients_with_contouring_issues))

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main():
    main_segmentation_map()
    main_overview_segmentation_map(data_dir_segmentation_map=cfg.save_dir_segmentation_map_dlc,
                                   source='DLC',
                                   save_file=cfg.filename_overview_structures_count_dlc_csv)


if __name__ == '__main__':
    main()

