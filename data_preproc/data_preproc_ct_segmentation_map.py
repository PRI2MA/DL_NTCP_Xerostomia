"""
TODO: enable `if patient_id not in invalid_uid_patient_ids:`
TODO: enable `assert False`

This script:
    - main_segmentation_map(): For each patient:
        - Load segmentation_map_citor.npy (if exists) and segmentation_map_dlc.npy (if exists).
        - For each relevant structure:
            - Check if the structure in segmentation_map_citor.npy is valid (i.e. count >= config.minimal_count)
                - If it is valid, add its segmentation to output_segmentation_mask.
                - If it is NOT valid, Check if the structure in segmentation_map_dlc.npy is valid.
                    - If it is valid, add its segmentation to output_segmentation_mask.
                    - Else it is not possible to have a segmentation of the structure anyways: continue.
        - Perform spacing_correction if cfg.perform_spacing_correction = True.
        - Save output_segmentation_map (segmentation_map.npy) to 'segmentation_map' folder.
        (TODO: save output_segmentation_map in DICOM_processed folder)
    - main_cropping_regions(): Determine the z/y/x upper and lower coordinates, using (potentially) spacing-corrected
        segmentation_map and our own segmentation_map value (see structures_values in config.py) of the structures.
        For each dimension z/y/x, determine upper and lower coordinates by filtering segmentation map voxels with
        value > 0 using:
            - z: parotid, crico and thyroid.
            - y: parotid.
            - x: parotid and submandibular.
        This function will output a cropping_regions.csv, which we can use to (manually) determine the desired bounding
        box size (z, y, x) (i.e. z/y/x cropping size). The z/y/x size of the bounding box will be used in
        data_preproc_dlc.py for cropping.

IMPORTANT NOTES:
    - main_segmentation_map(): every patient should at least have CITOR or DLC segmentation.
    - For every patient that has non-matching Frame of Reference UID CT and RTSTRUCT: ignore CITOR segmentation
        and use DLC segmentation, i.e. only use CITOR if patient_id not in invalid_uid_patient_ids.
    - overview_segmentation_map will be created for spacing corrected segmentation_map (if
        cfg.perform_spacing_correction = True), therefore the counts may be slightly different from those in
        overview_segmentation_map_citor and overview_segmentation_map_dlc!
    - In data_preproc.py we apply 'perform_spacing_correction=not cfg.perform_spacing_correction', i.e.
        do not apply spacing correction in data_preproc.py if it has been applied here, and vice versa.
    - main_cropping_regions(): the upper and lower coordinates correspond to the defined spacing (see config.py)!
        More explicitly, we FIRST perform spacing_correction and THEN determine the upper and lower z/y/x coordinates.
    - main_cropping_regions(): the upper and lower coordinates correspond to the array, and NOT to the plotted figures.
        (FYI: the upper part of the array corresponds to the lower part of the figure, and vice versa.)
"""
import os
import time
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

import data_preproc_config as cfg
from data_preproc_functions import create_folder_if_not_exists, Logger, sort_human
from data_preproc_ct_segmentation_map_citor import add_segmentation_maps, main_overview_segmentation_map


def get_subset_segmentation_map(segmentation_map, structures, structures_value_dict, as_binary, logger):
    """
    TODO: deprecated function - currently not being used in the whole pipeline
    Create segmentation_map from the structures (one or more) in 'structures'.
    E.g., if structures = ['parotid_li', 'parotid_re'], then the output is a segmentation_map containing
    segmentation of both structure 'parotid_li' and 'parotid_re'.

    Args:
        segmentation_map (Numpy array): original segmentation_map with values 0 (background), ..., nr_structures.
        structures (list): e.g.: structures = ['parotid_li', 'parotid_re']
        structures_value_dict (dict): dictionary with keys = all the structures, and value = label_map_value in
            the segmentation_map
        as_binary (bool): whether to return binary segmentation_map or not (i.e. values)
        logger: Logger for saving prints

    Returns:
        segmentation_relevant_structures (Numpy array): segmentation_map, with corresponding value at the coordinates
            of the relevant structures, and 0 otherwise. This array has the same shape as segmentation
        values:
        counts:
    """
    # For each relevant structure: get value
    values_i = list()
    counts_i = list()
    for structure in structures:
        structure_value = structures_value_dict[structure]
        structure_count = np.sum(segmentation_map == structure_value)

        # The following validity check is actually redundant, because segmentation_map should already only contain
        # the valid structures
        # Only add structure_value if structure_count >= minimal_count, because only then we consider the
        # segmentation_map to be valid
        if structure_count >= cfg.minimal_count:
            values_i.append(structure_value)
            counts_i.append(structure_count)
        else:
            logger.my_print('\t{structure} has count = {count} < {minimal_count}'
                            .format(structure=structure,
                                    count=structure_count,
                                    minimal_count=cfg.minimal_count),
                            level='warning')

    # Initiate segmentation of relevant structures as Numpy array with all False
    fill_value = False if as_binary else 0
    segmentation_map_i = np.full(segmentation_map.shape, fill_value)
    for v in values_i:
        if as_binary:
            # False (from segmentation initialization) + True (from equalizing to value) = True
            segmentation_map_i += (segmentation_map == v)
        else:
            segmentation_map_i += (segmentation_map == v) * v
    segmentation_map_i = segmentation_map_i.astype(int)

    return segmentation_map_i, values_i, counts_i


def spacing_correction(arr, metadata, out_spacing, is_label):
    """
    Change the voxel spacing of arr to out_spacing. Use the relevant metadata for this resampling.

    sitk.ResampleImageFilter(): examples:
        https://www.programcreek.com/python/example/123390/SimpleITK.ResampleImageFilter

    sitk.Transform(): a generic transformation. Can represent any of the SimpleITK transformations, and a composite
        transformation (stack of transformations concatenated via composition, last added, first applied).
        Source: https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/22_Transforms.html

    sitk.Get/SetDefaultPixelValue() (NOT TO CONFUSE WITH PixelIDValue!): get/set the pixel value when a transformed
        pixel is outside of the image. The default default pixel value is 0.
        Source: https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ResampleImageFilter.html

    Modification of:
    https://blog.tensorflow.org/2018/07/an-introduction-to-biomedical-image-analysis-tensorflow-dltk.html

    Args:
        arr (Numpy array): array with shape (z, y, x), for which we want to change its voxel spacing
        metadata (dict): meta of the reference image/arr, usually from CT
        out_spacing: desired output voxel spacing
        is_label: whether the itk_image is a label/segmentation_map. In that case we should use NearestNeighbor
            interpolation, because otherwise we would introduce new/decimal values.

    Returns:

    """
    # Initialize variables
    original_direction = metadata['Direction']
    original_origin = metadata['Origin']
    original_spacing = metadata['Spacing']

    # Map array to SITK image
    itk_image = sitk.GetImageFromArray(arr)
    itk_image.SetDirection(original_direction)
    itk_image.SetOrigin(original_origin)
    itk_image.SetSpacing(original_spacing)

    # Determine output shape after
    original_size = itk_image.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(original_direction)
    resample.SetOutputOrigin(original_origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(np.float64(arr.min()))

    # Source: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/21_Transforms_and_Resampling.html
    # sitkNearestNeighbor: is used to interpolate labeled images representing a segmentation, it is the only
    # interpolation approach which will not introduce new labels into the result.
    # sitkLinear: is used for most interpolation tasks, a compromise between accuracy and computational efficiency.
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5
        resample.SetInterpolator(sitk.sitkBSpline)  # sitk.sitkLinear

    # Perform spacing conversion
    resampled_itk_image = resample.Execute(itk_image)

    # Convert SITK Image back to array
    resampled_arr = sitk.GetArrayFromImage(resampled_itk_image)

    return resampled_arr


def get_segmentation_mask(segmentation_map, values):
    """
    Get segmentation mask (i.e. binary) with same shape as input segmentation_map.

    Args:
        segmentation_map (Numpy array): segmentation map
        values (list):

    Returns:

    """
    # Initialize mask
    mask = None
    for v in values:
        mask_v = (segmentation_map == v)
        if mask is None:
            mask = mask_v
        else:
            mask += mask_v

    return mask


def main_segmentation_map():
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    data_dir_ct = cfg.save_dir_ct
    save_root_dir = cfg.save_root_dir
    data_dir_segmentation_citor = cfg.save_dir_segmentation_map_citor
    data_dir_segmentation_dlc = cfg.save_dir_segmentation_map_dlc
    save_dir_segmentation_map = cfg.save_dir_segmentation_map
    minimal_count = cfg.minimal_count
    keep_segmentation_map_elements = cfg.keep_segmentation_map_elements
    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_ct_segmentation_map_logging_txt))
    start = time.time()

    # Create folder if not exist
    create_folder_if_not_exists(save_dir_segmentation_map)

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

        # Get patient_ids with no match between CT UID and RTSTRUCT UID
        invalid_uid = pd.read_csv(os.path.join(save_root_dir, cfg.filename_invalid_uid_csv), sep=';', index_col=0)
        invalid_uid = invalid_uid[invalid_uid['No Frame of Reference UID match CT RTSTRUCT']]
        # Convert filtered Pandas column to list, and convert patient_id = 111766 (type=int, because of Excel/csv file)
        # to patient_id = '0111766' (type=str)
        invalid_uid_patient_ids = ['%0.{}d'.format(cfg.patient_id_length) % x for x in invalid_uid.index]
        logger.my_print('Patient_ids with no match between CT UID and RTSTRUCT UID: {}'.format(invalid_uid_patient_ids))

        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

        for patient_id in tqdm(patients_list):
            logger.my_print('Patient_id: {}'.format(patient_id))

            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            # Load segmentation_map_citor.npy (if exists) and segmentation_map_dlc.npy (if exists)
            segmentation_map_citor = None
            # TODO: enable "if patient_id not in invalid_uid_patient_ids:".
            #  IMPORTANT: The if-statement should only be applied for CITOR segmentation_map, because
            #  if patient_id in invalid_uid_patient_ids (i.e. invalid FoR UID CT RTSTRUCT), then we should use DLC.
            # if patient_id not in invalid_uid_patient_ids:
            try:
                segmentation_map_citor = np.load(os.path.join(data_dir_segmentation_citor, patient_id,
                                                              cfg.filename_segmentation_map_npy))
            except:
                logger.my_print('Patient has no CITOR segmentation')


            segmentation_map_dlc = None
            try:
                segmentation_map_dlc = np.load(os.path.join(data_dir_segmentation_dlc, patient_id,
                                                            cfg.filename_segmentation_map_npy))
            except:
                logger.my_print('Patient has no DLC segmentation')

            # Make sure that at least one segmentation map exists
            assert (segmentation_map_citor is not None) or (segmentation_map_dlc is not None)

            # For each relevant and valid structure: store its segmentation_map in output_segmentation_map with value from
            # own value map
            output_segmentation_map = None
            if segmentation_map_citor is not None:
                output_segmentation_map = np.zeros_like(segmentation_map_citor)
            elif segmentation_map_dlc is not None:
                output_segmentation_map = np.zeros_like(segmentation_map_dlc)

            for structure in cfg.structures_uncompound_list:
                structure_value = cfg.structures_values[structure]

                # Check if structure in segmentation_map_citor.npy is valid (i.e. count >= config.minimal_count)
                # Else: Check if structure in segmentation_map_dlc.npy is valid
                if segmentation_map_citor is None:
                    structure_count_citor = minimal_count - 1
                else:
                    structure_count_citor = np.sum(segmentation_map_citor == structure_value)
                if segmentation_map_dlc is None:
                    structure_count_dlc = minimal_count - 1
                else:
                    structure_count_dlc = np.sum(segmentation_map_dlc == structure_value)

                # 'segmentation_map_citor/dlc is not None' is redundant, but is added for extra certainty
                if (segmentation_map_citor is not None) and (structure_count_citor >= minimal_count):
                    # output_segmentation_map += (segmentation_map_citor == structure_value) * structure_value
                    output_segmentation_map = add_segmentation_maps(
                        segmentation_map=output_segmentation_map, to_be_added_segmentation_map=segmentation_map_citor,
                        original_value=structure_value, new_value=structure_value,
                        keep_segmentation_map_elements=keep_segmentation_map_elements, logger=logger)
                elif (segmentation_map_dlc is not None) and (structure_count_dlc >= minimal_count):
                    # output_segmentation_map += (segmentation_map_dlc == structure_value) * structure_value
                    output_segmentation_map = add_segmentation_maps(
                        segmentation_map=output_segmentation_map, to_be_added_segmentation_map=segmentation_map_dlc,
                        original_value=structure_value, new_value=structure_value,
                        keep_segmentation_map_elements=keep_segmentation_map_elements, logger=logger)
                elif segmentation_map_dlc is not None:
                    logger.my_print('DLC (and possibly also CITOR) has no valid segmentation for structure = {}!'.
                                    format(structure), level='warning')
                # The following 'else'-statement should never be possible if the data collection has been done
                # correctly, because we should at least have CITOR or DLC
                else:
                    logger.my_print('There is no valid segmentation for structure = {}!'.
                                    format(structure), level='warning')
                    # TODO: enable "assert False"
                    # assert False

            # Perform spacing_correction
            # Note: in data_preproc.py we apply 'perform_spacing_correction=not cfg.perform_spacing_correction', i.e.
            # do not apply spacing correction in data_preproc.py if it has been applied here, and vice versa.
            if cfg.perform_spacing_correction:
                # Load ct_metadata
                ct_metadata_path = os.path.join(data_dir_ct, patient_id, cfg.filename_ct_metadata_json)
                json_file = open(ct_metadata_path)
                ct_metadata = json.load(json_file)
                logger.my_print('\tct_metadata["Spacing"][::-1] (input): {}'.format(ct_metadata["Spacing"][::-1]))
                logger.my_print('\tcfg.spacing[::-1] (output): {}'.format(cfg.spacing[::-1]))

                # Perform spacing_correction
                output_segmentation_map = spacing_correction(arr=output_segmentation_map, metadata=ct_metadata,
                                                             out_spacing=cfg.spacing,
                                                             is_label=True)
                logger.my_print('\toutput_segmentation_map.shape (spacing corrected): {}'
                                .format(output_segmentation_map.shape))
                logger.my_print('\toutput_segmentation_map.dtype (spacing corrected): {}'
                                .format(output_segmentation_map.dtype))
                json_file.close()

            # Make sure that the list of unique values in output_segmentation_map is less than or identical to the \
            # expected list of unique values, i.e. [0, 1, ..., len(cfg.structures_uncompound_list)]. +1 is for background
            # Normally those lists should be identical, but patient_id = 3573430 has no crico, esophagus_cerv and
            # thyroid, neither in CITOR nor in DLC.
            if not np.all(
                np.unique(output_segmentation_map) == [x for x in range(1 + len(cfg.structures_uncompound_list))]):
                logger.my_print('np.unique(output_segmentation_map): {}'.format(np.unique(output_segmentation_map)),
                                level='warning')
                logger.my_print('[x for x in range(1 + len(cfg.structures_uncompound_list))]: {}'.format(
                    [x for x in range(1 + len(cfg.structures_uncompound_list))]), level='warning')

            # Save results
            save_path = os.path.join(save_dir_segmentation_map, patient_id)
            create_folder_if_not_exists(save_path)

            # Save output_segmentation_map as Numpy array
            logger.my_print('output_segmentation_map.shape: {}'.format(output_segmentation_map.shape))
            np.save(file=os.path.join(save_path, cfg.filename_segmentation_map_npy), arr=output_segmentation_map)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main_cropping_regions():
    """
    Create overview of cropping region for each patient.

    Returns:

    """
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    data_dir_segmentation_map = cfg.save_dir_segmentation_map
    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_ct_segmentation_map_cropping_logging_txt))
    nr_of_decimals = cfg.nr_of_decimals
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

        # Save cropping information for each patient
        cropping_regions_dict = dict()

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

            # Load spacing-corrected segmentation_map
            segmentation_map = np.load(os.path.join(data_dir_segmentation_map, patient_id,
                                                    cfg.filename_segmentation_map_npy))

            # Cropping region
            # z-axis: we consider the most upper and most lower voxels (in z-dim)
            # (DEPRECATED): ... using parotid, crico, thyroid and mandible segmentation structures
            z_values_filter = [cfg.structures_values[x] for x in
                               cfg.parotis_structures + cfg.lower_structures + cfg.mandible_structures]
            # z_values_filter = [cfg.structures_values[x] for x in cfg.structures_uncompound_list]
            # mask.shape = (z, y, x)
            z_mask = get_segmentation_mask(segmentation_map=segmentation_map, values=z_values_filter)
            assert len(z_mask.shape) == 3
            z_upper = np.where(z_mask > 0)[0].max()
            z_lower = np.where(z_mask > 0)[0].min()

            z_size = z_upper - z_lower
            assert z_size >= 0

            # y-axis: we consider the most upper and most lower voxels (in y-dim)
            # (DEPRECATED): ... using parotid segmentation structures
            y_values_filter = [cfg.structures_values[x] for x in cfg.parotis_structures]
            # y_values_filter = [cfg.structures_values[x] for x in cfg.structures_uncompound_list]
            y_mask = get_segmentation_mask(segmentation_map=segmentation_map, values=y_values_filter)
            assert len(y_mask.shape) == 3
            y_upper = np.where(y_mask > 0)[1].max()
            y_lower = np.where(y_mask > 0)[1].min()

            y_size = y_upper - y_lower
            assert y_size >= 0

            # x-axis: we consider the most left and most right voxels (in x-dim)
            # (DEPRECATED): ... using parotid and submandibular segmentation structures
            x_values_filter = [cfg.structures_values[x] for x in cfg.parotis_structures + cfg.submand_structures]
            # x_values_filter = [cfg.structures_values[x] for x in cfg.structures_uncompound_list]
            x_mask = get_segmentation_mask(segmentation_map=segmentation_map, values=x_values_filter)
            assert len(x_mask.shape) == 3

            x_upper = np.where(x_mask > 0)[2].max()
            x_lower = np.where(x_mask > 0)[2].min()
            x_size = x_upper - x_lower
            assert x_size >= 0

            # Create cropping region information
            cropping_regions_i_dict = dict()
            z_num_slices = segmentation_map.shape[-3]
            y_num_rows = segmentation_map.shape[-2]
            x_num_columns = segmentation_map.shape[-1]
            cropping_regions_i_dict['z_num_slices'] = z_num_slices
            cropping_regions_i_dict['y_num_rows'] = y_num_rows
            cropping_regions_i_dict['x_num_columns'] = x_num_columns
            cropping_regions_i_dict['z_upper'] = z_upper
            cropping_regions_i_dict['z_lower'] = z_lower
            cropping_regions_i_dict['y_upper'] = y_upper
            cropping_regions_i_dict['y_lower'] = y_lower
            cropping_regions_i_dict['x_upper'] = x_upper
            cropping_regions_i_dict['x_lower'] = x_lower
            cropping_regions_i_dict['z_size'] = z_size
            cropping_regions_i_dict['y_size'] = y_size
            cropping_regions_i_dict['x_size'] = x_size
            cropping_regions_i_dict['z_upper_percentile'] = round(z_upper / z_num_slices, nr_of_decimals)
            cropping_regions_i_dict['z_lower_percentile'] = round(z_lower / z_num_slices, nr_of_decimals)
            cropping_regions_i_dict['y_upper_percentile'] = round(y_upper / y_num_rows, nr_of_decimals)
            cropping_regions_i_dict['y_lower_percentile'] = round(y_lower / y_num_rows, nr_of_decimals)
            cropping_regions_i_dict['x_upper_percentile'] = round(x_upper / x_num_columns, nr_of_decimals)
            cropping_regions_i_dict['x_lower_percentile'] = round(x_lower / x_num_columns, nr_of_decimals)

            # Save patient's cropping region information to all patients' dictionary
            cropping_regions_dict[patient_id] = cropping_regions_i_dict

        # Save cropping_regions_dict as csv file
        cropping_regions_dict_df = pd.concat([pd.DataFrame(v, index=[k]) for k, v in cropping_regions_dict.items()], axis=0)
        cropping_regions_dict_df.index.name = 'patient_id'
        cropping_regions_dict_df.to_csv(os.path.join(save_root_dir, cfg.filename_cropping_regions_csv), sep=';')

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main():
    main_segmentation_map()
    main_overview_segmentation_map(data_dir_segmentation_map=cfg.save_dir_segmentation_map,
                                   source='any',
                                   save_file=cfg.filename_overview_structures_count_csv)
    main_cropping_regions()


if __name__ == '__main__':
    main()
    
