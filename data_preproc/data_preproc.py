"""
This scripts prepares the dataset folder for modelling.
    - Arrays: CT, RTDOSE, segmentation_maps
    - Features: baseline xerostomia score ('baseline'), mean doses contralateral parotid gland ('D_Contra')

The arrays can be prepared as follows (in this order, if enabled):
    - perform_spacing_correction: change the spacing of the array, e.g. from (z, y ,x) spacing of (2, 1, 1) mm to
        (2, 2, 2) mm.
    - perform_cropping: crop a piece of the array, using bounding box coordinates. The bounding box should contain
        the relevant structures (see structures_compound_list in config.py).
    - perform_clipping: clip too large/low values between upper/lower bound, respectively.
    - perform_transformation: e.g. resize to model's desired input shape.
These actions and their hyperparameters can be enabled and disabled in config.py.

Cropping is done as follows:
    - For each of z/y/x: determine the upper and lower bounds (done in data_preproc_ct_segmentation_map_dlc.py).
    - Using the upper and lower bounds, we can determine the center:
        center = lower + (upper - lower) // 2 = upper - (upper - lower) // 2.
    - Perform slicing: arr[z/y/x_center - z/y/x_size // 2 : z/y/x_center + z/y/x_size // 2].
    - If slicing exceeds the array, i.e. z/y/x_center - z/y/x_size // 2 < 0 and/or
        z/y/x_center + z/y/x_size // 2 > num_slices/num_columns/num_rows, then pad the array with its minimum value
        (i.e. arr.min()). As a result, the cropped array will always have shape (z_size, y_size, x_size).

For each patient: the saved Numpy array with shape (c, z, y, x) = (num_channels, num_slices, num_rows, num_columns) is
a concatenation of (in this order):
    - CT array (shape: (1, z, y, x)).
    - RTDOSE array (shape: (1, z, y, x)).
    - Segmentation_map array (shape: (1, z, y, x)) from the main segmentation folder.

The CT values are clipped to values [-1000 HU, +1000 HU], which represent practical lower and upper limits of
the HU scale, corresponding to the radiodensities of air and bone respectively (DenOtter and Schubert, 2019).
Water has a radiodensity of 0 HU and most tissues are in the range -600 and +100 HU (Lambert et al., 2014).
Source: https://www.sciencedirect.com/science/article/pii/S1361841520302218

IMPORTANT NOTES:
    - We apply 'perform_spacing_correction=not cfg.perform_spacing_correction', i.e. we do not apply
        spacing correction here if it has been applied in data_preproc_ct_segmentation_map.py, and vice versa.

TODO:
- Dataset + DataLoader: we could/should use DatasetFolder, which basically is the underlying class of ImageFolder.

https://docs.monai.io/en/latest/transforms.html?highlight=loadimage#loadimage
Current default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader), (npz, npy -> NumpyReader),
    (others -> ITKReader).

Dataset folder structure:
    - 0:
        - patient_id_1_label0.npy
        ...
        - patient_id_m0_label0.npy
    ...
    - n:
        - patient_id_1_label1.npy
        ...
        - patient_id_m1_label1.npy
"""
import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from monai.transforms import Compose, Resize

import data_preproc_config as cfg
from data_preproc_functions import copy_file, copy_folder, create_folder_if_not_exists, Logger
from data_preproc_ct_segmentation_map import spacing_correction


def get_center(upper, lower):
    """
    For a single dimension: obtain the center between the upper and lower coordinate values.

    Args:
        upper (int): upper coordinate value
        lower (int): lower coordinate value

    Returns:
        center (int)
    """
    return lower + (upper - lower) // 2


def get_bounding_box_bounds(center, size):
    """
    For single dimension: get bounding box' upper and lower bound.

    Args:
        center (int): center coordinate value of the single dimension
        size (int): total length of the bounding box in the single dimension

    Returns:
        bb_upper (int): upper bound coordinate of bounding box
        bb_lower (int): lower bound coordinate of bounding box

    """
    # Add (size % 2 > 0) = 1 if size is not divisible by 2
    bb_upper = (center + size // 2 + (size % 2 > 0))
    bb_lower = (center - size // 2)
    return int(bb_upper), int(bb_lower)


def cropping(arr, cropping_region, z_size, y_size, x_size):
    """
    Crop relevant region of array, using the array coordinates of certain segmentation_map. For each dimension:
        - Determine the center using the distance between the upper and lower bounds. The
            upper and lower bounds are determined using the segmentation_map.
        - The half length of the bounding box in the dimension is z/y/x_size / 2: z/y/x_size should ideally (but not
            necessarily) be divisible by 2 for symmetry. Add/subtract this half length to/from the center
            from the previous step to respectively obtain the upper and lower array coordinates of the bounding box.
        - If the crop exceeds the array itself (i.e. the slicing range of the array is larger than the array itself),
            then pad the excess with arr.min() values. This is done separately for the upper and lower part of the crop.

    Note: the script also works if any of z/y/x_size is too small (i.e. not every segmentation_map will completely be
        in the cropped array). Therefore by setting z/y/x_size too low, we may not crop all relevant regions.
        The z/y/x_size should be the same for all patient, to distinguish people with different sizes.

    Args:
        arr (Numpy array): array to be cropped, with shape (z, y, x)
        cropping_region (dict or pandas.core.series.Series): containing for each dimension the upper and lower
            cropping coordinate
        z_size: desired output array's z-axis length, and ideally (but not necessarily) divisible by 2 for symmetry
        y_size: desired output array's x-axis length, and ideally (but not necessarily) divisible by 2 for symmetry
        x_size: desired output array's y-axis length, and ideally (but not necessarily) divisible by 2 for symmetry

    Returns:
        Numpy array of shape (z, y, x) = (z_size, y_size, x_size)
    """
    # Determine upper and lower z/y/x-coordinate values
    z_upper = int(cropping_region['z_upper'])
    z_lower = int(cropping_region['z_lower'])

    y_upper = int(cropping_region['y_upper'])
    y_lower = int(cropping_region['y_lower'])

    x_upper = int(cropping_region['x_upper'])
    x_lower = int(cropping_region['x_lower'])

    # Determine z/y/x-center coordinate values
    z_center = get_center(upper=z_upper, lower=z_lower)
    y_center = get_center(upper=y_upper, lower=y_lower)
    x_center = get_center(upper=x_upper, lower=x_lower)

    # Shift y-center
    # If y_center_shift_perc > 0, then the bounding box will be placed on higher y-indices of the array. (i.e. lower
    # image part, because the left-top corner index is (y, x) = (0, 0) and right-bottom corner index is
    # (y, x) = (num_rows, num_columns). Vice versa for y_center_shift_perc < 0. If y_center_shift_perc = 0, then
    # y_center will not be shifted
    y_center_shift = cfg.y_center_shift_perc * int(cropping_region['y_num_rows'])
    y_center += int(y_center_shift)

    # Determine bounding box' z/y/x upper and lower bounds using z/y/x centers and z/y/x sizes
    bb_z_upper, bb_z_lower = get_bounding_box_bounds(center=z_center, size=z_size)
    bb_y_upper, bb_y_lower = get_bounding_box_bounds(center=y_center, size=y_size)
    bb_x_upper, bb_x_lower = get_bounding_box_bounds(center=x_center, size=x_size)

    # Determine whether the crop exceeds the array itself
    # Upper part: if bb_z/y/x_upper > arr.shape[0/1/2], then concatenate array with values arr.min() and size
    # bb_z/y/x_upper - z/y/x_upper to 'arr'
    # Lower part: if bb_z/y/x_lower < 0, then concatenate array with values arr.min() and size
    # abs(bb_z_lower) = -1 * bb_z_lower to 'arr'
    z_padding_upper_size = bb_z_upper - arr.shape[0]
    y_padding_upper_size = bb_y_upper - arr.shape[1]
    x_padding_upper_size = bb_x_upper - arr.shape[2]

    # Cropping array with shape (z, y, x) to (z', y', x')
    arr = arr[max(0, bb_z_lower):bb_z_upper,
              max(0, bb_y_lower):bb_y_upper,
              max(0, bb_x_lower):bb_x_upper]

    # Pad upper if necessary
    padding_fill_value = arr.min()
    if z_padding_upper_size > 0:
        z_padding_upper_arr = np.full((z_padding_upper_size, arr.shape[1], arr.shape[2]), padding_fill_value)
        arr = np.concatenate((arr, z_padding_upper_arr), axis=0)
    if y_padding_upper_size > 0:
        y_padding_upper_arr = np.full((arr.shape[0], y_padding_upper_size, arr.shape[2]), padding_fill_value)
        arr = np.concatenate((arr, y_padding_upper_arr), axis=1)
    if x_padding_upper_size > 0:
        x_padding_upper_arr = np.full((arr.shape[0], arr.shape[1], x_padding_upper_size), padding_fill_value)
        arr = np.concatenate((arr, x_padding_upper_arr), axis=2)

    # Pad lower if necessary
    if bb_z_lower < 0:
        z_padding_lower_arr = np.full((abs(bb_z_lower), arr.shape[1], arr.shape[2]), padding_fill_value)
        arr = np.concatenate((z_padding_lower_arr, arr), axis=0)
    if bb_y_lower < 0:
        y_padding_lower_arr = np.full((arr.shape[0], abs(bb_y_lower), arr.shape[2]), padding_fill_value)
        arr = np.concatenate((y_padding_lower_arr, arr), axis=1)
    if bb_x_lower < 0:
        x_padding_lower_arr = np.full((arr.shape[0], arr.shape[1], abs(bb_x_lower)), padding_fill_value)
        arr = np.concatenate((x_padding_lower_arr, arr), axis=2)

    # Make sure the size is correct
    assert arr.shape == (z_size, y_size, x_size)

    return arr


def transformation(arr, resize_mode):
    """
    Perform transformation on input array.

    Args:
        arr (Numpy array): shape = [c, z, y, x] = [num_channels, num_slices, num_rows, num_columns], where usually c = 1
        resize_mode (str): resize mode for transformation, e.g. 'trilinear' (e.g. for CT and RTDOSE) and 'nearest'
            (for labels in segmentation maps)

    Returns:

    """
    xform = Compose([
        Resize(cfg.input_size, mode=resize_mode)
    ])
    return xform(arr)


def get_single_channel_array(arr_path, metadata, is_label, cropping_region, bb_size,
                             perform_spacing_correction, perform_cropping, perform_clipping, perform_transformation,
                             logger, dtype=None):
    """
    Load array in arr_path and perform data processing steps. This function contains basic transformations for Numpy
    arrays with shape (z, y, x):
        - (z, y, x)-spacing_correction
        - cropping subset area using bounding box size
        - value clipping
        - transformation (e.g. resizing)

    Args:
        arr_path (str): array path, including filename
        metadata (dict): containing metadata of physical space that is required for spacing_correction
        is_label (bool): whether input array contains labels (True, e.g. segmentation map) or not (False, e.g. CT and
            RTDOSE)
        cropping_region (dict or pandas.core.series.Series): containing for each dimension the upper and lower
            cropping coordinate
        bb_size (tuple): bounding box (/cropping) shape = (z, y, x) = [num_slices, num_rows, num_columns]
        perform_spacing_correction (bool): whether to perform spacing_correction (True) or not (False)
        perform_cropping (bool): whether to perform cropping (True) or not (False)
        perform_clipping (bool): whether to perform value clipping (True) or not (False)
        perform_transformation (bool): whether to perform transformation (True) or not (False)
        logger: Logger for saving prints
        dtype (str): (optional) conversion of datatype, e.g. 'int16'

    Returns:
        Numpy array with shape (1, z, y, x)
    """
    # Load arr with shape (num_slices, num_columns, num_rows)
    arr = np.load(arr_path)
    if dtype:
        arr = arr.astype(dtype)
    logger.my_print('\tarr.shape (initial): {}'.format(arr.shape))
    logger.my_print('\tarr.dtype (initial): {}'.format(arr.dtype))

    # Perform spacing_correction
    if perform_spacing_correction:
        arr = spacing_correction(arr=arr, metadata=metadata, out_spacing=cfg.spacing, is_label=is_label)
        logger.my_print('\tarr.shape (space corrected): {}'.format(arr.shape))
        logger.my_print('\tarr.dtype (space corrected): {}'.format(arr.dtype))

    # Perform cropping
    if perform_cropping:
        arr = cropping(arr=arr, cropping_region=cropping_region,
                       z_size=bb_size[0], y_size=bb_size[1], x_size=bb_size[2])
        logger.my_print('\tarr.shape (cropped): {}'.format(arr.shape))
        logger.my_print('\tarr.dtype (cropped): {}'.format(arr.dtype))

    # Perform value clipping
    if perform_clipping:
        # Clip CT pixel values (in Hounsfields Units (HU)) to [-1000 HU, +1000 HU]
        # Source: https://www.sciencedirect.com/science/article/pii/S1361841520302218
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
        arr = np.clip(arr, a_min=cfg.ct_min, a_max=cfg.ct_max)
        logger.my_print('\tarr.shape (clipped): {}'.format(arr.shape))
        logger.my_print('\tarr.dtype (clipped): {}'.format(arr.dtype))

    # Add channel dimension
    arr = np.expand_dims(arr, axis=0)

    # Perform transformations
    if perform_transformation:
        # Determine resize mode based on whether the input exists of labels or not
        if is_label:
            resize_mode = cfg.label_resize_mode
        else:
            resize_mode = cfg.non_label_resize_mode
        arr = transformation(arr=arr, resize_mode=resize_mode)
        logger.my_print('\tarr.shape (transformed): {}'.format(arr.shape))
        logger.my_print('\tarr.dtype (transformed): {}'.format(arr.dtype))

    return arr


def main_array():
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    save_dir_ct = cfg.save_dir_ct
    save_dir_rtdose = cfg.save_dir_rtdose
    save_dir_segmentation_map = cfg.save_dir_segmentation_map
    save_dir_dataset_full = cfg.save_dir_dataset_full
    patient_id_col = cfg.patient_id_col
    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_array_logging_txt))
    bb_size = cfg.bb_size
    start = time.time()

    # Create folder if not exist
    create_folder_if_not_exists(save_dir_dataset_full)

    for _, d_i in zip(mode_list, data_dir_mode_list):
        # TODO: currently redundant
        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        patients_list_ct = os.listdir(save_dir_ct)
        patients_list_rtdose = os.listdir(save_dir_rtdose)
        patients_list_segmentation_map = os.listdir(save_dir_segmentation_map)
        assert patients_list_ct == patients_list_rtdose == patients_list_segmentation_map

        patients_list = patients_list_ct
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        # Load cropping regions of all patients
        if cfg.perform_cropping:
            cropping_regions = pd.read_csv(os.path.join(save_root_dir, cfg.filename_cropping_regions_csv),
                                           sep=';', index_col=0)
            # Convert filtered Pandas column to list, and convert patient_id = 111766 (type=int, because of Excel/csv file)
            # to patient_id = '0111766' (type=str)
            cropping_regions_index_name = cropping_regions.index.name
            cropping_regions.index = ['%0.{}d'.format(patient_id_length) % x for x in cropping_regions.index]
            cropping_regions.index.name = cropping_regions_index_name

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

            # Load ct_metadata, for performing spacing_correction
            ct_metadata = None
            if cfg.perform_spacing_correction:
                ct_metadata_path = os.path.join(save_dir_ct, patient_id, cfg.filename_ct_metadata_json)
                json_file = open(ct_metadata_path)
                ct_metadata = json.load(json_file)
                logger.my_print('\tct_metadata["Spacing"][::-1] (input): {}'.format(ct_metadata["Spacing"][::-1]))
                logger.my_print('\tcfg.spacing[::-1] (output): {}'.format(cfg.spacing[::-1]))

            # Extract cropping region
            cropping_region_i = None
            if cfg.perform_cropping:
                cropping_region_i = cropping_regions.loc[patient_id].to_dict()

            # Load and preprocess CT
            logger.my_print('\t----- CT -----')
            ct_arr_path = os.path.join(save_dir_ct, patient_id, cfg.filename_ct_npy)
            ct_arr = get_single_channel_array(arr_path=ct_arr_path, dtype=None, metadata=ct_metadata, is_label=False,
                                              cropping_region=cropping_region_i,
                                              bb_size=bb_size,
                                              perform_spacing_correction=cfg.perform_spacing_correction,
                                              perform_cropping=cfg.perform_cropping,
                                              perform_clipping=cfg.perform_clipping,
                                              perform_transformation=cfg.perform_transformation, logger=logger)

            # Load and preprocess RTDOSE
            # Note: "can't convert np.ndarray of type numpy.uint16. The only supported types are: float64,
            # float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool."
            logger.my_print('\t----- RTDOSE -----')
            resampled_rtdose_arr_path = os.path.join(save_dir_ct, patient_id, cfg.filename_rtdose_npy)
            resampled_rtdose_arr = get_single_channel_array(arr_path=resampled_rtdose_arr_path, dtype='int16',
                                                            metadata=ct_metadata, is_label=False,
                                                            cropping_region=cropping_region_i,
                                                            bb_size=bb_size,
                                                            perform_spacing_correction=cfg.perform_spacing_correction,
                                                            perform_cropping=cfg.perform_cropping,
                                                            perform_clipping=False,
                                                            perform_transformation=cfg.perform_transformation,
                                                            logger=logger)

            # Load and preprocess segmentation_map
            # Spacing_correction is not necessary, because this is already performed in main_segmentation_map() in
            # data_preproc_ct_segmentation_map.py
            # Note: we apply 'perform_spacing_correction=not cfg.perform_spacing_correction', i.e. we do not apply
            # spacing correction here if it has been applied in data_preproc_ct_segmentation_map.py, and vice versa.
            logger.my_print('\t----- Segmentation_map -----')
            segmentation_map_arr_path = os.path.join(save_dir_segmentation_map, patient_id,
                                                     cfg.filename_segmentation_map_npy)
            segmentation_map_arr = get_single_channel_array(arr_path=segmentation_map_arr_path, dtype='int16',
                                                            metadata=ct_metadata, is_label=True,
                                                            cropping_region=cropping_region_i,
                                                            bb_size=bb_size,
                                                            perform_spacing_correction=not cfg.perform_spacing_correction,
                                                            perform_cropping=cfg.perform_cropping,
                                                            perform_clipping=False,
                                                            perform_transformation=cfg.perform_transformation,
                                                            logger=logger)

            # Save as Numpy array
            save_dir_dataset_full_i = os.path.join(save_dir_dataset_full, patient_id)
            create_folder_if_not_exists(save_dir_dataset_full_i)
            np.save(file=os.path.join(save_dir_dataset_full_i, 'ct.npy'), arr=ct_arr)
            np.save(file=os.path.join(save_dir_dataset_full_i, 'rtdose.npy'), arr=resampled_rtdose_arr)
            np.save(file=os.path.join(save_dir_dataset_full_i, 'segmentation_map.npy'), arr=segmentation_map_arr)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main_dataset():
    """
    Copy-paste .npy files to dataset folder.

    Returns:

    """
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    save_dir_dataset_full = cfg.save_dir_dataset_full
    # TODO: temporary
    save_dir_dataset = cfg.save_dir_dataset
    # Taste
    # save_dir_dataset_full = '//zkh/appdata/RTDicom/PRI2MA/MDACC/preprocessed/dataset_full_mdacc'
    # save_dir_dataset = 'D:/MyFirstData_MDACC/dataset_mdacc_taste'
    # cfg.filename_endpoints_csv = 'taste_features_MDACC.csv'
    # cfg.endpoint = 'HN35_Taste_M06'
    # Dyspaghia:
    # save_dir_dataset_full = '//zkh/appdata/RTDicom/PRI2MA/MDACC/preprocessed/dataset_full_mdacc'
    # save_dir_dataset = 'D:/MyFirstData_MDACC/dataset_mdacc_dyshagia'
    # cfg.filename_endpoints_csv = 'dysphagia_features_MDACC.csv'
    # cfg.endpoint = 'Dysphagia_6MO'
    patient_id_col = cfg.patient_id_col
    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_features_logging_txt))
    start = time.time()

    # Create folder if not exist
    create_folder_if_not_exists(save_dir_dataset)

    for _, d_i in zip(mode_list, data_dir_mode_list):
        # TODO: currently redundant
        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # TODO: temporary
        # Load Excel file containing patient_id and their endpoints/labels/targets
        endpoints_csv = os.path.join(save_root_dir, cfg.filename_endpoints_csv)
        df = pd.read_csv(endpoints_csv, sep=';')
        # Taste:
        # df = pd.read_csv(os.path.join(save_root_dir, 'features.csv'), sep=';')
        # label_mapper = {'Helemaal niet': 0, 'Een beetje': 0, 'Nogal': 1, 'Heel erg': 1}
        # df['HN35_Taste_M06'] = df['HN35_Taste_M06'].map(label_mapper)
        if len(df.columns) == 1:
            df = pd.read_csv(endpoints_csv, sep=',')

        # Copy endpoints.csv to dataset folder
        src = endpoints_csv
        dst = os.path.join(save_dir_dataset, cfg.filename_endpoints_csv)
        copy_file(src, dst)

        # Create dictionary for the labels {label0: [patient_id1_label0, ..., patient_idm0_label0], ...,
        # labeln: [patient_id1_label1, ..., patient_idm1_label1]}
        labels_patients_dict = dict()
        # Convert filtered Pandas column to list, and convert patient_id = 111766 (type=int, because of Excel/csv file)
        # to patient_id = '0111766' (type=str)
        # labels_patients_dict['0'] = ['%0.{}d'.format(patient_id_length) % x for x in
        #                              df[df[cfg.endpoint] <= 2][patient_id_col].tolist()]
        # labels_patients_dict['1'] = ['%0.{}d'.format(patient_id_length) % x for x in
        #                              df[df[cfg.endpoint] > 2][patient_id_col].tolist()]
        labels_patients_dict['0'] = ['%0.{}d'.format(patient_id_length) % x for x in
                                     df[df[cfg.endpoint] == 0][patient_id_col].tolist()]
        labels_patients_dict['1'] = ['%0.{}d'.format(patient_id_length) % x for x in
                                     df[df[cfg.endpoint] == 1][patient_id_col].tolist()]

        for label, patients_list in labels_patients_dict.items():
            save_dir_dataset_label = os.path.join(save_dir_dataset, str(label))
            create_folder_if_not_exists(save_dir_dataset_label)

            # Testing for a small number of patients
            if cfg.test_patients_list is not None:
                test_patients_list = cfg.test_patients_list
                patients_list = [x for x in test_patients_list if x in patients_list]

            for patient_id in tqdm(patients_list):
                logger.my_print('Patient_id: {id}'.format(id=patient_id))

                # TODO: currently redundant
                # Determine folder type of interest ('with_contrast'/'no_contrast')
                # TODO: temporary, for MDACC
                if use_umcg:
                    folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
                else:
                    folder_type_i = ''

                # Copy files from dataset_full to dataset/0 and dataset/1
                src = os.path.join(save_dir_dataset_full, patient_id)
                dst = os.path.join(save_dir_dataset_label, patient_id)
                copy_folder(src, dst)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main_features():
    """
    Extract features: for deep learning (see load_data.py) and logistic regression model (see main.py and load_data.py).

    Note: run before: split_stratified_sampling.ipynb to generate `splits_full.csv`, then create_endpoints_csv.ipynb
        to generate `endpoints.csv`. The file `endpoints.csv` is required for this function.

    Returns:

    """
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    save_dir_dataset = cfg.save_dir_dataset
    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_features_logging_txt))
    start = time.time()

    # Create folder if not exist
    create_folder_if_not_exists(save_dir_dataset)

    for _, d_i in zip(mode_list, data_dir_mode_list):
        # TODO: currently redundant
        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

            # Load Excel file containing patient_id, features and their endpoints/labels/targets
            df = pd.read_csv(os.path.join(save_root_dir, cfg.filename_endpoints_csv), sep=';')

            # Select features
            cols = [cfg.patient_id_col, cfg.endpoint, cfg.baseline_col] + cfg.features + cfg.ext_features
            df = df[cols]

            # TODO: temporary
            # save_dir_dataset = 'D:/Github/dl_ntcp_xerostomia/unfilled_holes/dataset'
            # Save features df as csv file
            for filename in [cfg.filename_features_csv, cfg.filename_stratified_sampling_csv]:
                df.to_csv(os.path.join(save_dir_dataset, filename), sep=';', index=False)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main():
    # main_array()
    main_dataset()
    # main_features()


if __name__ == '__main__':
    main()
