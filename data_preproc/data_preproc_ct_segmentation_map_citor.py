"""
This script:
    - main_segmentation_map(): Patients can have multiple RTSTRUCTs (i=0, 1, ...), each RTSTRUCT can have different
        structures: load and save each segmentation mask (segmentation_{i}.npy) and structures value map + count
        (structures_value_count_{i}.json (dictionary): for each relevant structure: {segmentation_map value, count}) to
        'segmentation_map_citor' folder.
    - main_best_segmentation_map(): Some patients have multiple RTSTRUCT folders and/or files: combine segmentation
        maps.
        - For each patient: for each relevant structure: consider the segmentation with the largest number of voxels in
            the segmentation_map. That is: for each patient:
            - Start with zeros Numpy array (with same shape as CITOR segmentation_map).
            - For each relevant structure:
                - Check which (of the multiple) segmentation has largest count (using structures_value_count_{i}.json).
                - Load the corresponding segmentation_map.
                - Add its voxels to initial zeros array with value from own consistent value map.
                - Save the combined segmentation_map (segmentation_map_citor.npy) in 'segmentation_map_citor' folder.

IMPORTANT NOTES:
    - The single (segmentation_map_{i}.npy), and therefore also the combined segmentation_map
        (segmentation_map_citor.npy), will only contain the relevant structures.
    - 'count' (in structures_value_count) indicates the number of voxels in the created segmentation_map. That is,
        if a structure has count = 0, then this structure is not contained in segmentation_map, but the structure
        could have > 0 voxels in the raw RTSTRUCT. Moreover, it either has voxels but were not considered
        (see if-statement in get_segmentation_map()), and/or the contours were invalid (see try-except
        in get_segmentation_map()). If a relevant structure has count = 0, then the structure really has no voxels
        in the segmentation and/or invalid contours.
    - Useful Github repository for extracting RTSTRUCTs: `RT-Utils` (https://github.com/qurit/rt-utils)
"""
import os
import cv2
import time
import json
import numpy as np
import pandas as pd
import pydicom as pdcm
from tqdm import tqdm
from glob import glob
from pydicom import dcmread
from skimage.draw import polygon
from PIL import Image, ImageDraw

import data_preproc_config as cfg
from data_preproc_functions import (copy_folder, create_folder_if_not_exists, get_all_folders, list_to_txt_str, Logger,
                                    round_nths_of_list, set_default, sort_human)


def read_ds(ds):
    """
    Extract meta-data from the input Pydicom DataSet (ds).

    Args:
        ds: RTSTRUCT as Pydicom DataSet

    Returns:
        contours (list): list of dictionaries, where keys = structure; values = meta-data of the corresponding structure
            - contours[i]: meta-data of i^{th} structure.
            - len(contours) = number of structures.
            - contours[i]['contours'] = list of contours per slice:
                len(contours[i]['contours']) == nr_slices with that contour.
            - contours[i]['contours'][j]: list [x1, y1, z1, x2, y2, z2, ...] of contour coordinates on a single
                slice.
            - Note: z1, z2, z3, ... have the same value.
            - Note 2: len(contours[0]['contours'][j]) is divisible by 3.
    """
    contours = []
    # Use Try because ROIContourSequence may not be available in ds
    try:
        for i in range(len(ds.ROIContourSequence)):
            contour = dict()
            try:
                contour['color'] = ds.ROIContourSequence[i].ROIDisplayColor
            except:
                contour['color'] = [255, 0, 0]
            contour['number'] = ds.ROIContourSequence[i].ReferencedROINumber
            contour['name'] = ds.StructureSetROISequence[i].ROIName

            # Make sure that the value in the segmentation map is consistent, i.e. make sure that
            # ds.ROIContourSequence[i].ReferencedROINumber == ds.StructureSetROISequence[i].ROINumber
            if contour['number'] == ds.StructureSetROISequence[i].ROINumber:
                # There is no guarantee that ContourData is present in a given item in the ROI Contour Sequence.
                # Source: https://github.com/pydicom/pydicom/issues/1206
                # contour['contours'] = [x.ContourData for x in ds.ROIContourSequence[i].ContourSequence]
                # Round every 3rd element (z-axis, slices) to one decimal
                contour['contours'] = [round_nths_of_list(in_list=x.ContourData, n=3, decimal=1)
                                       for x in ds.ROIContourSequence[i].ContourSequence]

            contours.append(contour)
    except:
        pass

    return contours


def fill_contour(unfilled_contour_map, structure_value):
    """
    For one structure and single slice: fill contours, but not holes.

    Source: https://stackoverflow.com/questions/37160143/how-can-i-extract-internal-contours-holes-with-python-opencv
    Source: https://java2blog.com/cv2-findcontours-python/

    Args:
        unfilled_contour_map: Numpy array with shape (y, x) = (num_rows, num_columns), only outline
            of contours (i.e. unfilled)
        structure_value: contour fill value

    Returns:
        segmentation_map_unfilled_hole: Numpy array with shape (y, x) = (num_rows, num_columns), filled contour,
            but not holes
    """
    img_i = cv2.normalize(src=unfilled_contour_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_8UC1)
    contours_i, hierarchy_i = cv2.findContours(img_i, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Consider holes
    holes_i = [contours_i[i] for i in range(len(contours_i)) if hierarchy_i[0][i][3] >= 0]

    # Fill contours, but not holes
    segmentation_map_unfilled_hole = cv2.fillPoly(img_i, holes_i, (255, 255, 255))
    segmentation_map_unfilled_hole = ((segmentation_map_unfilled_hole > 0) * structure_value).\
        astype(unfilled_contour_map.dtype)

    return segmentation_map_unfilled_hole


def get_segmentation_map(contours, slices, image, modify_structure, logger):
    """
    This function will return a segmentation_map (as Numpy array) for the different structures. This segmentation_map
    can be used for constructing the contours. Moreover, the segmentation_map contain filled contours for
    the relevant structures such as 'parotid_li', 'submandibular_li', etc.

    Source: http://aapmchallenges.cloudapp.net/forums/3/2/

    Args:
        contours (dict): contour information for all structures
        slices (list): list of CT slices, each as a Pydicom DataSet class
        image (Numpy array): CT slices as Numpy array with shape (x, y, z) (!) (note: usually a 3D Numpy array has
            shape = (z, y, x))
        modify_structure (function): mapping of input structure name to a consistent structure name, because
            different datasets (e.g. CITOR vs DLC) use different (input) structure names
        logger:

    Returns:
        segmentation_map: Numpy array with shape (z, y, x) = (num_slices, num_rows, num_columns). The array contain
            values 0 (background), 1, 2, ..., n, where n is the number of different contours (but only for relevant
            structures, if available). For example, if the value 6 corresponds to the structure 'parotid_re'. Then
            segmentation_map == 6 will return a (binary) mask array of that structure.
            Important note: the value for differs per person, i.e. for one person the value = 6 could correspond to
            'parotid_re', whereas for another person the value = 6 could correspond to 'myelum'!)
        colors: colors for each structure. This is only useful for plotting.
        structure_con_issues_list: list of structures for which there are issues of creating contours,
            caught in try-except (see function).
    """
    # Initiate a list of structures for which there are issues to create contours from
    structure_con_issues_list = list()

    # ImagePositionPatient: source: https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032
    # Round z-indices (slices) to one decimal
    z = [round(s.ImagePositionPatient[2], 1) for s in slices]  # e.g. z = [47.7, 50.2, 52.7, ...]
    pos_y = slices[0].ImagePositionPatient[1]
    spacing_y = slices[0].PixelSpacing[1]
    pos_x = slices[0].ImagePositionPatient[0]
    spacing_x = slices[0].PixelSpacing[0]

    # For every structure: create filled contour (and without filling holes if cfg.unfill_holes = True) and
    # add this filled contour to output_segmentation_map
    output_segmentation_map = np.zeros_like(image, dtype=np.uint8)
    for con in contours:
        num = int(con['number'])

        has_structure_con_issues_i = False

        # RTSTRUCTs could have structures (e.g. 'External') that overlaps with other structures. This overlap will
        # mess up the segmentation map, because it will replace the voxels of these other structures. Therefore
        # we only consider segmentation for relevant structures
        # Note: the function can deal with overlapping segmentations, but more overlap = more information loss
        segmentation_map_i = np.zeros_like(output_segmentation_map, dtype=output_segmentation_map.dtype)
        if modify_structure(structure=con['name'], logger=logger) in cfg.structures_uncompound_list:
            for c in con['contours']:
                # try:
                nodes = np.array(c).reshape((-1, 3))
                assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
                # nodes[0, 2] = z-coordinate in physical space (e.g. 52.7)
                # z_index = 2 if z = [47.7, 50.2, 52.7, ...] and nodes[0, 2] = 52.7
                z_index = z.index(nodes[0, 2])
                y = (nodes[:, 1] - pos_y) / spacing_y
                x = (nodes[:, 0] - pos_x) / spacing_x

                if cfg.unfill_holes:
                    # Extract contours only outline, so not filled. Will be filled when each contour of
                    # the structure has been extracted (see fill_contour() below)
                    yx = [(y, x) for y, x in zip(y, x)]
                    # Initiate empty image
                    image_segmentation_map_i = Image.fromarray(np.zeros_like(segmentation_map_i[:, :, z_index],
                                                                             dtype=np.uint8))
                    # Draw contour outline on image_segmentation_map
                    draw = ImageDraw.Draw(image_segmentation_map_i)
                    draw.polygon(yx, fill=0, outline=1)

                    # Add contour outline to segmentation_map_i
                    # Use mask, because contours of the same structure may still overlap!
                    mask = (segmentation_map_i[:, :, z_index] == 0)
                    contour = ((np.array(image_segmentation_map_i) > 0) * num).astype(segmentation_map_i.dtype)
                    segmentation_map_i[:, :, z_index] += mask * contour
                else:
                    # Fill contour + holes
                    xx, yy = polygon(x, y)
                    output_segmentation_map[xx, yy, z_index] = num
                # except:
                #     has_structure_con_issues_i = True

            # Fill outline, but not holes
            if cfg.unfill_holes:
                if len(np.unique(segmentation_map_i)) > 2:
                    print('np.unique(segmentation_map_i): {}'.format(np.unique(segmentation_map_i)))
                    assert False

                nr_slices = segmentation_map_i.shape[-1]
                for z_index in range(nr_slices):
                    # Only consider slices that contain the contours
                    if (segmentation_map_i[:, :, z_index] == num).sum() > 0:
                        # Fill contour map, but not holes
                        contour_map = fill_contour(unfilled_contour_map=segmentation_map_i[:, :, z_index],
                                                   structure_value=num)

                        # Make sure to not overlap different structures when adding
                        mask = (output_segmentation_map[:, :, z_index] == 0)
                        output_segmentation_map[:, :, z_index] += (mask * contour_map)

            # Save structure if there are issues with extracting the contours from
            if has_structure_con_issues_i:
                structure_con_issues_list.append(con['name'])

    # print('structure_con_issues_list:', structure_con_issues_list)
    colors = tuple(np.array([con['color'] for con in contours]) / 255.0)

    # Swap axes from (x, y, z) to (z, y, x)
    output_segmentation_map = np.swapaxes(output_segmentation_map, 0, 2)

    return output_segmentation_map, colors, structure_con_issues_list


def modify_structure_citor(structure, logger):
    """
    This script applies modifications to (try to) get consistent notations of the structure as possible,
    across different files.

    Args:
        structure (str): structure, e.g. 'Parotid_L'
        logger:
    Returns:

    """
    # Create dictionary for mapping structure. 'key' will be replaced by 'value' in 'structure'
    # BUG: some structure names in CITOR RTSTRUCT are (slightly) different, e.g. 'Cricopharyngeus' instead of
    # 'crico', 'Supraglotticlarynx' instead of 'supraglottic' or 'PMC_...' (incorrect) instead of 'PCM...' (correct)
    if cfg.use_umcg:
        source_replacement_dict = {
            'Parotid_L': 'parotis_li',
            'Parotid_R': 'parotis_re',
            'parotis_li': 'parotis_li',
            'parotis_re': 'parotis_re',
            'Submandibular_L': 'submandibularis_li',
            'Submandibular_R': 'submandibularis_re',
            'submandibularis_li': 'submandibularis_li',
            'submandibularis_re': 'submandibularis_re',
            'Crico': 'crico',
            'cricopharyngeus': 'crico',  # patient_id = 9066477
            'Esophagus_Cerv': 'esophagus_cerv',
            'Thyroid': 'thyroid',
            'BuccalMucosa_L': 'buccalmucosa_li',
            'BuccalMucosa_R': 'buccalmucosa_re',
            'GlotticArea': 'glotticarea',
            'Mandible': 'mandible',
            'OralCavity_Ext': 'oralcavity_ext',
            'PCM_Inf': 'pcm_inf',
            'PCM_Med': 'pcm_med',
            'PCM_Sup': 'pcm_sup',
            'PMC_Inf': 'pcm_inf',
            'PMC_Med': 'pcm_med',
            'PMC_Sup': 'pcm_sup',
            'Supraglottic': 'supraglottic',
        }
    else:
        # TODO: temporary, for MDACC
        source_replacement_dict = {
            'Oral_cavity': 'oralcavity_ext',
            'cavity_oral': 'oralcavity_ext',
            'Extended_oral_cavity': 'oralcavity_ext',

            'Parotid_L': 'parotis_li',
            'Lt_parotid_gland': 'parotis_li',
            'Left_parotid': 'parotis_li',

            'Parotid_R': 'parotis_re',
            'Rt_parotid_gland': 'parotis_re',
            'Right_parotid': 'parotis_re',

            'Glnd_submand_L': 'submandibularis_li',
            'Lt_Submandibular_Gland': 'submandibularis_li',

            'Glnd_submand_R': 'submandibularis_re',
            'Rt_Submandibular_Gland': 'submandibularis_re',

            # 'BuccalMucosa_L': 'buccalmucosa_li',
            # 'BuccalMucosa_R': 'buccalmucosa_re',

            'SPC': 'pcm_sup',
            'Musc_Constric_S': 'pcm_sup',
            'Musc_Constrict_S': 'pcm_sup',
            'MPC': 'pcm_med',
            'Musc_Constric_M': 'pcm_med',
            'Musc_Constrict_M': 'pcm_med',
            'IPC': 'pcm_inf',
            'Musc_Constric_I': 'pcm_inf',
            'Musc_Constrict_I': 'pcm_inf',

            'cricopharyngeus': 'crico',
            'cricopharyngeal_muscle': 'crico',

            'Supraglottic_Larynx': 'supraglottic',
            'Larynx_SG': 'supraglottic',

            'Esophagus_U': 'esophagus_cerv',
            'Esophagus': 'esophagus_cerv',

            'Mandible': 'mandible',

            'Thyroid_cartilage': 'thyroid',

            'Glottic_Area': 'glotticarea',
        }


    # Replace manually defined structure name by consistent structure name (e.g. 'cricopharyngeus' to 'crico')
    consistent_structure_names = [v for k, v in source_replacement_dict.items() if k.lower() == structure.lower()]
    # If len(structures) == 0, then only apply lowercase to 'structure'
    if len(set(consistent_structure_names)) == 1:
        structure = consistent_structure_names[0]
    elif len(consistent_structure_names) > 1:
        logger.my_print('structure: {}'.format(structure))
        logger.my_print('consistent_structure_names: {}'.format(consistent_structure_names))
        assert False

    return structure.lower()


def add_segmentation_maps(segmentation_map, to_be_added_segmentation_map, original_value, new_value,
                          keep_segmentation_map_elements, logger):
    """
    Replace segmentation_map elements with new_value where to_be_added_segmentation_map == original_value.

    Note: elements of segmentation_map and to_be_added_segmentation_map may overlap. In the case of overlapping,
        the 'keep_segmentation_map_elements' argument determines whether we keep the elements from either
        segmentation_map (True) or from to_be_added_segmentation_map (False)

    Args:
        segmentation_map (Numpy array):
        to_be_added_segmentation_map (Numpy array): Numpy array with same shape as segmentation_map
        original_value (int): value of 'o_be_added_segmentation_map
        new_value (int): replacement of original_value from to_be_added_segmentation_map for segmentation_map
        keep_segmentation_map_elements (bool): if True, keep elements of segmentation_map. Else replace by elements
        of to_be_added_segmentation_map
        logger:

    Returns:
        segmentation_map (Numpy array): segmentation_map + elements from to_be_added_segmentation_map, where
            the newly added elements from to_be_added_segmentation_map have value = new_value.
    """
    # Initialize variables
    segmentation_map_dtype = segmentation_map.dtype

    # Add to_be_added_segmentation_map to segmentation_map, but where original_value of to_be_added_segmentation_map
    # will be replaced by new_value
    mask = to_be_added_segmentation_map == original_value

    # Note: original_value may not be available in to_be_added_segmentation_map (i.e. np.sum(mask) == 0), and
    # therefore segmentation_map will also not get new_value
    if np.sum(mask) > 0:
        # Get unique initial values of segmentation_map
        unique_values = np.unique(segmentation_map)

        # Make sure new_value is not in segmentation_map yet
        assert new_value not in unique_values

        # Keep elements of segmentation_map (i.e. only replace elements in segmentation_map that are still 0).
        # Else replace by elements of to_be_added_segmentation_map
        if keep_segmentation_map_elements:
            mask_background = mask * (segmentation_map == 0)
            segmentation_map += (mask_background * new_value).astype(segmentation_map_dtype)
        else:
            segmentation_map[mask] = new_value

        # Generally mask * (segmentation_map == 0) have overlap and thus non-zero voxels
        if np.sum(mask_background) > 0:
            unique_values_updated = np.unique(segmentation_map)
            unique_values_expected = np.sort(np.append(arr=unique_values, values=[new_value]))
            if not np.all(unique_values_updated == unique_values_expected):
                logger.my_print('unique_values_updated: {}'.format(unique_values_updated))
                logger.my_print('unique_values_expected: {}'.format(unique_values_expected))
                assert False
        else:
            unique_values_updated = np.unique(segmentation_map)
            if not np.all(unique_values_updated == unique_values):
                logger.my_print('unique_values_updated: {}'.format(unique_values_updated))
                logger.my_print('unique_values: {}'.format(unique_values))
                assert False

    return segmentation_map


def main_segmentation_map():
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    # data_dir_ct = cfg.data_dir
    # data_dir_citor = cfg.data_dir
    save_root_dir = cfg.save_root_dir
    save_dir = cfg.save_dir_segmentation_map_citor
    patient_folder_types = cfg.patient_folder_types
    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_ct_segmentation_map_citor_logging_txt))
    patients_with_contouring_issues = list()
    start = time.time()

    # Create folder if not exist
    create_folder_if_not_exists(save_dir)

    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir_ct = cfg.data_dir.format(d_i)
        data_dir_citor = data_dir_ct

        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # Get intersection(cohort, CITOR) patient_ids
        logger.my_print('Listing intersection(cohort, CITOR) patient ids and their files...')
        patients_citor_list = os.listdir(data_dir_citor)
        patients_citor_list = sort_human(patients_citor_list)
        n_patients = len(patients_citor_list)
        logger.my_print('Total number of CITOR patients: {n}'.format(n=n_patients))

        # Initialize dataframe
        df = pd.DataFrame(columns=['Patient_id', 'Filename'])

        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_citor_list = [x for x in test_patients_list if x in patients_citor_list]

        for patient_id in tqdm(patients_citor_list):
            logger.my_print('Patient_id: {id}'.format(id=patient_id))

            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            # Get all folders and sub-folders
            all_folders_ct = get_all_folders(os.path.join(data_dir_ct, patient_id, folder_type_i))
            all_folders_citor = get_all_folders(os.path.join(data_dir_citor, patient_id, folder_type_i))

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
            image = np.stack([s.pixel_array for s in slices], axis=-1)  # axis = -1: image.shape = (x, y, z)

            # Get RTSTRUCT folder(s)
            # Normally patients should have one RTSTRUCT folder and one file, but sometimes they have multiple. Consider
            # all of them
            path_rtss = [x for x in all_folders_citor if x.endswith('\\RTSTRUCT')]
            n_path_rtss = len(path_rtss)

            # 'with_contrast' has highest priority: if it does not contain one or more of the required folders,
            # then search in 'no_contrast'
            # TODO: temporary, for MDACC
            if use_umcg:
                if n_path_rtss == 0:
                    all_folders_citor_2 = get_all_folders(os.path.join(data_dir_citor, patient_id, patient_folder_types[1]))
                    path_rtss = [x for x in all_folders_citor_2 if x.endswith('\\RTSTRUCT')]
                    n_path_rtss = len(path_rtss)
                elif n_path_rtss > 1:
                    logger.my_print('Patient {id} has {n} RTSTRUCT folders!'.format(id=patient_id, n=n_path_rtss),
                                    level='warning')

            # Patients could have multiple RTSTRUCT folders: use for-loop
            assert n_path_rtss >= 1
            i = 0
            for p in path_rtss:
                files_rtss = glob(p + '/*')
                n_files_rtss = len(files_rtss)
                # Patients can have multiple RTSTRUCT files. Consider all of them
                if n_files_rtss > 1:
                    logger.my_print('Patient {id} has {n} RTSTRUCT files!'.format(id=patient_id, n=n_files_rtss),
                                    level='warning')

                # Normally patients should have one RTSTRUCT file, but sometimes they have multiple. Consider all of them
                for f_rtss in files_rtss:
                    logger.my_print('Loading RTSTRUCT file: {f}'.format(f=f_rtss))

                    # Load RTSTRUCT
                    ds = dcmread(f_rtss)

                    # Extract contour information from structures (if available)
                    contours = read_ds(ds)

                    # Extract segmentation_map and potentially a list of structures for which there are issues with
                    # extracting contours
                    segmentation_map, colors, structure_con_issues_i = get_segmentation_map(
                        contours=contours, slices=slices, image=image, modify_structure=modify_structure_citor,
                        logger=logger)

                    # Create structures_value_count_{i}.json
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

                        # Initiate structure_dict
                        structure_dict = dict()

                        # Check if structure_value (0, 1, ..., n) is in segmentation_map, because segmentation_map may not
                        # contain certain structures!
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

                        # Construct structure value-count dictionary for the given structure
                        count = int(count)
                        structure_dict['Value'] = value
                        structure_dict['Count'] = count
                        structure_dict['Modified_name'] = modify_structure_citor(structure=structure, logger=logger)
                        structures_value_count[structure] = structure_dict

                    # Save results
                    save_path = os.path.join(save_dir, patient_id)
                    create_folder_if_not_exists(save_path)

                    # Save structures_value_count
                    with open(os.path.join(save_path, cfg.filename_structures_value_count_i_json.format(i=i)), 'w') as f:
                        json.dump(structures_value_count, f, default=set_default)

                    # Save segmentation_map as Numpy array
                    logger.my_print('segmentation_map.shape: {}'.format(segmentation_map.shape))
                    np.save(file=os.path.join(save_path, cfg.filename_segmentation_map_i_npy.format(i=i)),
                            arr=segmentation_map)

                    # Save structures for which there are contouring issues
                    if len(structure_con_issues_i) > 0:
                        patients_with_contouring_issues.append(patient_id + '_{i}'.format(i=i))
                        with open(os.path.join(save_path,
                                               cfg.filename_structures_with_contouring_issues_citor_i_txt.format(i=i)),
                                  'w') as f:
                            f.write(list_to_txt_str(structure_con_issues_i))

                    # Append df with filename
                    df_i = pd.DataFrame({'Patient_id': patient_id + '_{i}'.format(i=i),
                                         'Filename': f_rtss.replace(os.path.join(data_dir_citor, patient_id), '')}, index=[0])
                    df = pd.concat([df, df_i])

                    i += 1

        # Save list of patient_ids for which there is at least one structure with contouring issues
        with open(os.path.join(save_root_dir, cfg.filename_patients_with_contouring_issues_citor_txt), 'w') as f:
            f.write(list_to_txt_str(patients_with_contouring_issues))

        # Save df with filename
        df.to_csv(os.path.join(save_root_dir, cfg.filename_patient_ids_with_filename_csv), index=False)
        # df.to_csv('//zkh/appdata/RTDicom/Hendrike Neh/Data collection/Hung/patient_ids_with_filename.csv', index=False)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main_best_segmentation_map():
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    data_dir_segmentation_map = cfg.save_dir_segmentation_map_citor
    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_ct_segmentation_map_best_logging_txt))
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = cfg.data_dir.format(d_i)

        # TODO: currently redundant
        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # Get all patient_ids
        logger.my_print('Listing all patient ids and their files...')
        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

        for patient_id in tqdm(patients_list):
            logger.my_print('Patient_id: {id}'.format(id=patient_id))

            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            # Get all files
            search_path = os.path.join(data_dir_segmentation_map, patient_id)
            all_patient_files = os.listdir(search_path)
            all_structures_value_count_files = [x for x in all_patient_files if
                                                (x.startswith('structures_value_count_') and x.endswith('.json'))]

            # Dictionary: for each relevant structure: get the best segmentation file
            best_segmentation_map_dict = dict()
            # For each relevant structure: determine best segmentation map = segmentation map with largest counts of voxels
            # in segmentation_map and add its segmentation to output_segmentation_map
            output_segmentation_map = None
            for structure in cfg.structures_uncompound_list:

                best_count = -1
                best_file = None
                for structures_value_count_file in all_structures_value_count_files:
                    # Load structures_value_count_{i}.json file
                    json_filename_with_path = open(os.path.join(search_path, structures_value_count_file))
                    structures_value_count_dict = json.load(json_filename_with_path)
                    structure_counts_list = [v['Count'] for _, v in structures_value_count_dict.items() if
                                             v['Modified_name'] == structure]
                    # Make sure that structures_value_count_dict contains at most one Count for one structure
                    # assert len(structure_counts_list) <= 1

                    # Note: sum of an empty list = 0 (i.e. sum([]) = 0)
                    count_i = sum(structure_counts_list)
                    if best_count < count_i:
                        best_count = count_i
                        best_file = structures_value_count_file

                # Store file with best count
                best_segmentation_map_dict[structure] = {'Best_count': best_count, 'Best_file': best_file}

                # Determine best `i' (i = 0, 1, ..., nr_rtstruct_files)
                best_json_filename_with_path = open(os.path.join(search_path, best_file))
                best_structures_value_count_dict = json.load(best_json_filename_with_path)
                best_i = int(best_file.replace('structures_value_count_', '').replace('.json', ''))
                best_segmentation_map_i = np.load(
                    os.path.join(search_path, cfg.filename_segmentation_map_i_npy.format(i=best_i)))

                # TODO: redundant, because best_count and best_file are determined above
                # If a RTSTRUCT file contains multiple v['Modified_name'] == structure,
                # then select the structure_value with highest count
                # if len(best_structure_value_count_list) >= 2:
                #     best_count = max([x[1] for x in best_structure_value_count_list])
                #     best_structure_value_count_list = [x[0] for x in best_structure_value_count_list if
                #                                        (x[1] == best_count) and (x[1] > 0)]

                # Combine segmentation_maps
                # Initialize segmentation_map, if not yet
                if output_segmentation_map is None:
                    output_segmentation_map = np.zeros_like(best_segmentation_map_i)

                # Add best structure's voxels to output_segmentation_map array with value from own value map
                # Determine structure_value of best `i'
                best_structure_value_count_list = [[v['Value'], v['Count']]
                                                   for _, v in best_structures_value_count_dict.items() if
                                                   (v['Modified_name'] == structure)]

                # Sometimes RTSTRUCT contains multiple modified_name == structure. Especially MDACC has sometimes RTSTUCT
                # files for which a certain modified_name structure is present multiple times in the RTSTRUCT files:
                # E.g. patient_id = 1406457912 has 'Cavity_Oral' and 'ORAL_CAVITY'.
                # Fix this issue by considering the structure with most counts.
                if len(best_structure_value_count_list) > 1:
                    max_count = max([x[1] for x in best_structure_value_count_list])
                    best_structure_value_count_list = [x for x in best_structure_value_count_list if x[1] == max_count]

                # Note that the structure may not be present in any RTSTRUCT and hence len(best_structure_value_count_list) == 0
                if not len(best_structure_value_count_list) <= 1:
                    logger.my_print('best_structure_value_count_list: {}'.format(best_structure_value_count_list))
                    assert False
                new_structure_value = cfg.structures_values[structure]
                if len(best_structure_value_count_list) == 1:
                    original_structure_value = best_structure_value_count_list[0][0]
                    assert type(original_structure_value) == int

                    # output_segmentation_map += (best_segmentation_map_i == original_structure_value) * new_structure_value
                    output_segmentation_map = add_segmentation_maps(
                        segmentation_map=output_segmentation_map, to_be_added_segmentation_map=best_segmentation_map_i,
                        original_value=original_structure_value, new_value=new_structure_value,
                        keep_segmentation_map_elements=cfg.keep_segmentation_map_elements, logger=logger)
                else:
                    logger.my_print('Patient has no segmentation of {structure}, i.e. the structure has either count = 0 '
                                    'or is not available in any RTSTRUCT file'.format(structure=structure), level='warning')

            # Save results
            save_path = search_path

            # Save best_segmentation_map_dict
            with open(os.path.join(save_path, cfg.filename_best_count_files_json), 'w') as f:
                json.dump(best_segmentation_map_dict, f, default=set_default)

            # Save output_segmentation_map
            logger.my_print('output_segmentation_map.shape: {}'.format(output_segmentation_map.shape))
            np.save(file=os.path.join(save_path, cfg.filename_segmentation_map_npy), arr=output_segmentation_map)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main_overview_all_segmentation_map():
    """
    Create csv: for each patient: count number of occurrences for each structure

    Returns:

    """
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    data_dir_segmentation_map = cfg.save_dir_segmentation_map_citor
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = cfg.data_dir.format(d_i)

        # TODO: currently redundant
        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # Get all patient_ids
        print('Listing all patient ids and their files...')
        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)
        n_patients = len(patients_list)
        print('Total number of patients: {n}'.format(n=n_patients))

        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

        df = pd.DataFrame()
        for patient_id in tqdm(patients_list):
            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            # Get all files
            search_path = os.path.join(data_dir_segmentation_map, patient_id)
            all_patient_files = os.listdir(search_path)
            all_structures_value_count_files = [x for x in all_patient_files if
                                                (x.startswith('structures_value_count_') and x.endswith('.json'))]

            for structure_value_count_file in all_structures_value_count_files:
                # Load JSON
                json_file = open(os.path.join(search_path, structure_value_count_file))
                structure_value_count_dict = json.load(json_file)

                # Note: we may get duplicated v['Modified_name'] (e.g. 2x external for patient_id = 0885661), but this
                # will raise an error by pd.concat(). However at least one of them (in fact: both of them) has
                # v['Count'] == 0.
                df_i = pd.concat([pd.DataFrame(v, index=[k]) for k, v in structure_value_count_dict.items() if v['Count'] > 0], axis=0)

                # In rare cases we have multiple rows with same Modified_name. Consider the one with highest count
                # Low values at the top (e.g. A and 0), high values at the bottom (e.g. Z and 9999)
                df_i = df_i.sort_values(by=['Modified_name', 'Count'], ascending=True)
                df_i = df_i.drop_duplicates(subset=['Modified_name'], keep='last')

                # Create dataframe
                column_names = df_i['Modified_name']
                df_i = pd.DataFrame(df_i['Count']).T
                idx = int(structure_value_count_file.replace('structures_value_count_', '').replace('.json', ''))
                df_i.columns = column_names
                df_i.index = ['{}_{}'.format(patient_id, idx)]
                # Concatenate dataframes
                df = pd.concat([df, df_i])

        # Sort columns based on their names, and save df to csv file
        df = df.sort_index(axis=1)
        df.to_csv(os.path.join(save_root_dir, cfg.filename_overview_all_structures_count_citor_csv), sep=';')

    end = time.time()
    print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    print('DONE!')


def main_overview_segmentation_map(data_dir_segmentation_map, source, save_file):
    """
    Create csv: for each patient: count (best) number of occurrences for each structure using segmentation_map.npy.

    Returns:

    """
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    structures_uncompound_list = cfg.structures_uncompound_list
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = cfg.data_dir.format(d_i)

        # TODO: currently redundant
        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # Get all patient_ids
        print('Listing all patient ids and their files...')
        patients_list = os.listdir(data_dir)
        patients_list = sort_human(patients_list)
        n_patients = len(patients_list)
        print('Total number of patients: {n}'.format(n=n_patients))

        df = pd.DataFrame()
        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

        for patient_id in tqdm(patients_list):
            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            # Get all files
            search_path = os.path.join(data_dir_segmentation_map, patient_id)
            try:
                all_patient_files = os.listdir(search_path)
            except:
                all_patient_files = []
                print('Patient {id} has no {src} segmentation'.format(id=patient_id, src=source))

            segmentation_map_file = [x for x in all_patient_files if x == cfg.filename_segmentation_map_npy]
            assert len(segmentation_map_file) <= 1

            df_i_dict = dict()
            if len(segmentation_map_file) == 1:
                # Load segmentation_map.npy
                segmentation_map_file = segmentation_map_file[0]
                segmentation_map = np.load(os.path.join(data_dir_segmentation_map, patient_id, segmentation_map_file))
                for structure in structures_uncompound_list:
                    structure_count = np.sum(segmentation_map == cfg.structures_values[structure])
                    df_i_dict[structure] = structure_count

            else:
                for structure in structures_uncompound_list:
                    df_i_dict[structure] = 0

            df_i = pd.DataFrame(df_i_dict, index=[patient_id])
            df = pd.concat([df, df_i])

        # Sort columns based on their names, and save df to csv file
        df = df.sort_index(axis=1)
        df.to_csv(os.path.join(save_root_dir, save_file), sep=';')

    end = time.time()
    print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    print('DONE!')


def main_patients_get_dlc():
    """
    TEMPORARY FUNCTION, FOR SIMPLICITY. CONTAINS MANUAL OPERATIONS. IS NOT COMPLETELY LINKED WITH config.py.

    This function copy-paste cfg.data_dir's patient CT folders to local folder. Those folders should be uploaded
    to Mirada, in order to obtain their DLCs.

    Returns:

    """
    # Initialize variables
    data_dir = '//zkh/appdata/RTDicom/HungChu/A1_CT_IBM/All_Correct_Selection/DICOM/'
    save_dir = 'D:/DICOM/MyFirstData/DICOM_DLC'

    create_folder_if_not_exists(save_dir)

    # Open txt file with all patients for which we need to retreive DLC
    patients_list = os.listdir(data_dir)
    patients_list = sort_human(patients_list)
    n_patients = len(patients_list)
    print('Total number of patients: {n}'.format(n=n_patients))

    # Testing for a small number of patients
    if cfg.test_patients_list is not None:
        test_patients_list = cfg.test_patients_list
        patients_list = [x for x in test_patients_list if x in patients_list]

    for patient_id in tqdm(patients_list):
        print('patient_id:', patient_id)
        src = os.path.join(data_dir, patient_id)
        # e.g.: src = '//zkh/appdata/RTDicom/HungChu/A1_CT_IBM/All_Correct_Selection/DICOM/01122334'
        # Get CT path name
        path_ct = get_all_folders(src)
        path_ct = [x for x in path_ct if x.endswith('\\CT')]
        assert len(path_ct) == 1
        folder_name = path_ct[0].replace(data_dir, '')
        # e.g. folder_name: '01122334\\20130102\\CT'
        dst = os.path.join(save_dir, folder_name)
        # e.g. dst: 'D:/DICOM/MyFirstData/DICOM_DLC/01122334\\20130102\\CT'

        try:
            copy_folder(src=src, dst=dst)
        except:
            print('src = {src} cannot be moved to dst = {dst}'.format(src=src, dst=dst))


def main():
    # main_segmentation_map()
    main_best_segmentation_map()
    main_overview_all_segmentation_map()
    main_overview_segmentation_map(data_dir_segmentation_map=cfg.save_dir_segmentation_map_citor,
                                   source='',
                                   save_file=cfg.filename_overview_structures_count_citor_csv)

    # Only run separately, and manually: first create 'patients_dlc.txt' using 'patients_in_citor_and_dlc.txt'
    # 'patients_without_valid_citor_rtstruct.txt'
    # main_patients_get_dlc()


if __name__ == '__main__':
    main()
