"""
Data preparation for CT and RTDOSE.

For each patient in data_dir:
    - Load RTDOSE DICOM file.
    - Extract basic meta-data.
    - Load RTDOSE pixel array.
    - Load CT pixel array.
    - Match RTDOSE to CT grid using Resampling from SimpleITK, in order to overlay RTDOSE and CT.
    - Extract meta-data from before and after data processing.
    - Store meta-data in JSON file.
    - Save RTDOSE (in cGy unit using DoseGridScaling * 100) and CT (in Hounsfield Units) pixel array as Numpy array
        with shape (z, y, x) = (num_slices, num_rows, num_columns).

IMPORTANT NOTES:
    # Source (wrong?): https://github.com/rachellea/ct-volume-preprocessing/blob/f466a9416c7de0fcd078623faf6944c2c5046e40/preprocess_volumes.py#L440
    # Source: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/03_Image_Details.html
    - (sitk) image.GetSize() returns: (x, y, z) = (num_columns, num_rows, num_slices). Note, arr.shape returns
    (z, y, x) automatically if 'arr' was obtained from sitk.GetArrayFromImage(image) (because 'arr' is a Numpy array)!
    - (pydicom) ds.pixel_array.shape returns: (z, y, x) = (num_slices, num_rows, num_columns).
    - Z-spacing value is not always (correctly) filled in the DICOM header. Therefore we compute the z-spacing
        ourselves by taking the upper and lower z-coordinate and divide this difference by: number of slices - 1.
    - SITK already converts the pixel data to Hounsfield Units (HU):
        # https://github.com/pydicom/pydicom/blob/da556e33b/pydicom/pixel_data_handlers/util.py#L207
        # https://github.com/pydicom/pydicom/issues/1125
        # https://github.com/jonasteuwen/SimpleITK-examples/blob/master/examples/apply_lut.py
        import SimpleITK as sitk
        from pydicom import dcmread
        from pydicom.pixel_data_handlers.util import apply_modality_lut

        # SITK
        arr_ct = sitk.GetArrayFromImage(image_ct)

        # Pydicom
        # Get minimum and maximum value of each CT slice
        mins_i = []
        maxs_i = []

        ct_files = glob(path_ct + '/*')
        for f in ct_files:
            ds = dcmread(f)
            arr_i = ds.pixel_array
            # 'ModalityLUTSequence' in ds
            hu_arr_i = apply_modality_lut(arr_i, ds)
            mins_i.append(hu_arr_i.min())
            maxs_i.append(hu_arr_i.max())

        assert arr_ct.min() == min(mins_i)
        assert arr_ct.max() == max(maxs_i)

data_dir should have the following structure:
data_dir:
    - folder_patient_id_0
        - folder_random_name
            - CT
                - file_random_name_0.dcm
                - file_random_name_1.dcm
                ...
                - file_random_name_X.dcm
            - RTDOSE
                - file_random_name.dcm
            - RTSTRUCT
                - file_random_name.dcm
            - folder_random_name
    - ...
    - folder_patient_id_X
        - folder_random_name
            - CT
                - file_random_name_0.dcm
                - file_random_name_1.dcm
                ...
                - file_random_name_X.dcm
            - RTDOSE
                - file_random_name.dcm
            - RTSTRUCT
                - file_random_name.dcm
            - folder_random_name
"""
import os
import time
import json
import numpy as np
import pandas as pd
import pydicom as pdcm
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from pydicom import dcmread

import data_preproc_config as cfg
from data_preproc_functions import create_folder_if_not_exists, get_all_folders, Logger, set_default, sort_human


def load_ct(path_ct, logger):
    """
    Load CT DICOM files in the path as an SITK Image. The pixel intensity of the raw CT array 'x' will
    be converted to Hounsfield Units (HU) by: y = x * slope + intercept.

    Args:
        path_ct: folder path containing the CT DICOM files. E.g. path_rtdose = D:/DICOM/0111766/20130221/CT/
        logger:

    Returns:
        image_ct: SITK Image with SITKs shape (x, y, z) = (num_columns, num_rows, num_slices)
    """
    # reader = sitk.ImageSeriesReader()
    # dicom_names = reader.GetGDCMSeriesFileNames(path_ct)
    # reader.SetFileNames(dicom_names)
    # image_ct = reader.Execute()
    # Load CT slices and sort them by position
    slices = [pdcm.read_file(dcm) for dcm in glob(path_ct + '/*')]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    arr_ct = np.stack([s.pixel_array for s in slices], axis=0)  # axis = 0: image.shape = (z, y, x)

    # Construct metadata_ct
    metadata_ct = json.loads(slices[0].to_json())  # slices[0]: bottom slice
    direction = metadata_ct['00200037']['Value'] + [0, 0, 1]  # Image Orientation (Patient)
    # number_of_pixel_components = metadata_ct['00280002']['Value']  # Samples per Pixel
    origin = metadata_ct['00200032']['Value']  # Image Position (Patient)

    # Get (x, y, z) spacing
    xy_spacing = metadata_ct['00280030']['Value']

    # Determine z-spacing by minimum and maximum z-coordinate value
    # Assumption: z-spacing value is constant
    z_min_max = [slices[0].ImagePositionPatient[2], slices[-1].ImagePositionPatient[2]]
    z_spacing = (z_min_max[-1] - z_min_max[0]) / (len(slices) - 1)
    z_spacing = round(z_spacing, 1)
    # if '00180088' in metadata_ct.keys():
    #     # TODO: temporary, for MDACC
    #     # MDACC sometimes have non-constant but consistent distance between slices, e.g. 3, 2, 3, 2, etc.
    #     # Then the DICOM tag '00180088' (Spacing Between Slices) value = -2.5
    #     z_spacing = abs(metadata_ct['00180088']['Value'][0])
    # elif (int(metadata_ct['00200013']['Value'][0]) == 1) and ('00201002' in metadata_ct.keys()):
    #     if metadata_ct['00201002']['Value'][0] == 0:
    #         logger.my_print("metadata_ct['00201002']['Value'][0] = 0. "
    #                         "Will be replaced by arr_ct.shape[0] = {}.".format(arr_ct.shape[0]), level='warning')
    #         metadata_ct['00201002']['Value'][0] = arr_ct.shape[0]
    #     z_spacing = metadata_ct['00180050']['Value'][0] / (arr_ct.shape[0] / metadata_ct['00201002']['Value'][0])
    # elif int(metadata_ct['00200013']['Value'][0]) != 1:
    #     if metadata_ct['00200013']['Value'][0] == 0:
    #         logger.my_print("metadata_ct['00200013']['Value'][0] = 0. "
    #                         "Will be replaced by arr_ct.shape[0] = {}.".format(arr_ct.shape[0]), level='warning')
    #         metadata_ct['00200013']['Value'][0] = arr_ct.shape[0]
    #     z_spacing = metadata_ct['00180050']['Value'][0] / (arr_ct.shape[0] / metadata_ct['00200013']['Value'][0])
    # else:
    #     z_spacing = metadata_ct['00180050']['Value'][0]

    # TODO: temporary
    print('z_spacing:', z_spacing)
    assert z_spacing > 0
    spacing = xy_spacing + [z_spacing]

    # Make sure that z-spacing value is in the correct range
    if (spacing[-1] < 0.5) or (spacing[-1] > 5):
        logger.my_print('Spacing = {}'.format(spacing), level='warning')
    # spacing = Pixel Spacing (x, y) + slice_thickness_loaded (z), where slice_thickness_loaded =
    # Slice Thickness / (num_slices_loaded / num_slices_at_slice_thickness)
    # E.g. num_slices_loaded = 402 (i.e. arr_ct.shape = (402, 512, 512)), but num_slices_at_slice_thickness = 201 at
    # Slice Thickness = 2 mm (i.e. there are 201 slices at z_spacing = 2 mm), so then there are 402 slices at:
    # slice_thickness_loaded = 2 mm / (402 / 201) = 2 / 2 = 1 mm

    # Convert pixel intensity of x to Hounsfield Units (HU): y = x * slope + intercept
    slope = metadata_ct['00281053']['Value'][0]  # commonly: 1
    intercept = metadata_ct['00281052']['Value'][0]  # commonly: -1024
    arr_ct = arr_ct * slope + intercept
    # arr_ct.dtype: uint16 (pixel intensity: [0, ...]) --> float64 (HU: [-1024.0, ...]) --> int16 (HU: [-1024, ...])
    arr_ct = arr_ct.astype(cfg.ct_dtype)

    # Convert to Image
    image_ct = sitk.GetImageFromArray(arr_ct)
    image_ct.SetOrigin(origin)
    image_ct.SetDirection(direction)
    image_ct.SetSpacing(spacing)

    return image_ct, metadata_ct


def load_rtdose(path_rtdose):
    """
    Load RTDOSE DICOM file in a folder as an SITK Image.

    Args:
        path_rtdose: folder containing the RTDOSE DICOM file. E.g. path_rtdose = 'D:/DICOM/0111766/20130221/RTDOSE/'

    Returns:
        image_rtdose: SITK Image with SITKs shape (x, y, z) = (num_columns, num_rows, num_slices)
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path_rtdose)
    reader.SetFileNames(dicom_names)
    image_rtdose = reader.Execute()
    image_rtdose = image_rtdose[:, :, :, -1]
    return image_rtdose


def load_rtdose_file(file_rtdose):
    """
    Load RTDOSE DICOM file as an SITK Image.

    Args:
        file_rtdose: RTDOSE DICOM file. E.g. file_rtdose = 'D:/DICOM/0111766/20130221/RTDOSE/DS1.dcm'

    Returns:
        image_rtdose: SITK Image with SITKs shape (x, y, z) = (num_columns, num_rows, num_slices)
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(file_rtdose)
    image_rtdose = reader.Execute()
    return image_rtdose


def resample_rtdose_as_ct(image_rtdose, image_ct):
    """
    Resample RTDOSE's Image to CT's reference image grid.

    Args:
        image_rtdose: SITK image_rtdose with SITKs shape (x, y, z) = (num_columns, num_rows, num_slices)
        image_ct: SITK image_ct with SITKs shape (x, y, z) = (num_columns, num_rows, num_slices)

    Returns:
        The Numpy arrays from sitk.GetArrayFromImage() will have automatically their shape transformed from
        (x, y, z) = (num_columns, num_rows, num_slices) to (z, y, x)!
    """
    # Resample image_rtdose in the same grid as image_ct (reference)
    # defaultPixelValue (NOT TO CONFUSE WITH PixelID / PixelIDValue!): get/set the pixel value when a transformed pixel
    # is outside of the image. The default pixel value is 0.
    # Source: https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ResampleImageFilter.html
    # Source: https://discourse.itk.org/t/resample-to-same-spacing-and-size-and-align-to-same-origin/1378/8
    # Source: https://discourse.itk.org/t/resample-volume-to-specific-voxel-spacing-simpleitk/3531/2
    # Source: https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5
    # RTDOSE: (126, 76, 30)
    # CT: (512, 512, 100)
    resampled_image_rtdose = sitk.Resample(image_rtdose, image_ct,
                                           transform=sitk.Transform(),
                                           interpolator=sitk.sitkLinear,  # sitk.sitkBSpline,  # sitk.sitkLinear
                                           defaultPixelValue=0,
                                           outputPixelType=image_rtdose.GetPixelID())

    # Convert SITK Image to Numpy Array
    arr_rtdose = sitk.GetArrayFromImage(image_rtdose)
    resampled_arr_rtdose = sitk.GetArrayFromImage(resampled_image_rtdose)
    arr_ct = sitk.GetArrayFromImage(image_ct)

    # Check
    assert arr_ct.shape == resampled_arr_rtdose.shape

    return resampled_arr_rtdose, arr_ct, arr_rtdose, resampled_image_rtdose


def main():
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    # data_dir = cfg.data_dir
    save_root_dir = cfg.save_root_dir
    save_dir_rtdose = cfg.save_dir_rtdose
    save_dir_ct = cfg.save_dir_ct
    patient_folder_types = cfg.patient_folder_types
    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_ct_rtdose_logging_txt))
    # filename_rtdose = cfg.filename_rtdose_dcm
    start = time.time()

    # Create folder if not exist
    create_folder_if_not_exists(save_dir_rtdose)
    create_folder_if_not_exists(save_dir_ct)

    for _, d_i in zip(mode_list, data_dir_mode_list):
        data_dir = cfg.data_dir.format(d_i)

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
            logger.my_print('Patient_id: {}'.format(patient_id))

            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            #####
            logger.my_print('\tSTEP 1 (RTDOSE, CT) - Fetching folder and files...')
            # List all folders of the patient
            all_folders = get_all_folders(os.path.join(data_dir, patient_id, folder_type_i))
            # Get relevant path
            path_ct = [x for x in all_folders if x.endswith('\\CT')]
            path_rtdose = [x for x in all_folders if x.endswith('\\RTDOSE')]

            # 'with_contrast' has highest priority: if it does not contain one or more of the required folders,
            # then search in 'no_contrast'
            # TODO: temporary, for MDACC
            if use_umcg:
                if len(path_ct) == 0:
                    all_folders_2 = get_all_folders(os.path.join(data_dir, patient_id, patient_folder_types[1]))
                    path_ct = [x for x in all_folders_2 if x.endswith('\\CT')]
                if len(path_rtdose) == 0:
                    all_folders_2 = get_all_folders(os.path.join(data_dir, patient_id, patient_folder_types[1]))
                    path_rtdose = [x for x in all_folders_2 if x.endswith('\\RTDOSE')]

            # Checks
            assert len(path_ct) == len(path_rtdose) == 1
            path_ct = path_ct[0]
            path_rtdose = path_rtdose[0]

            # # Sometimes there are multiple RTDOSE folders and/or files per patient:
            # # Get all files in RTDOSEs. If there are multiple RTDOSE files, then select the file 'Echte_Original_plan.dcm'
            # # and the folder that contains this file.
            # all_rtdose_files = []
            # for p in path_rtdose:
            #     all_rtdose_files += glob(p + '/*')
            # if len(all_rtdose_files) == 1:
            #     path_rtdose_file = all_rtdose_files
            # else:
            #     path_rtdose_file = [x for x in all_rtdose_files if x.endswith(filename_rtdose)]
            #     path_rtdose = [x.replace('\\' + filename_rtdose, '') for x in path_rtdose_file]

            rtdose_file = os.listdir(path_rtdose)
            assert len(rtdose_file) == 1
            path_rtdose_file = os.path.join(path_rtdose, rtdose_file[0])

            #####
            logger.my_print('\tSTEP 2 (RTDOSE, CT) - Loading Pydicom Dataset (RTDOSE) and Images (RTDOSE, CT)...')
            ds = dcmread(path_rtdose_file)
            image_ct, _ = load_ct(path_ct=path_ct, logger=logger)
            image_rtdose = load_rtdose(path_rtdose)

            #####
            logger.my_print('\tSTEP 3 (RTDOSE, CT) - Extracting meta-data: general...')
            # CT
            ds_ct_dict = dict()
            ds_ct_dict['ID'] = patient_id
            ds_ct_dict['Path_ct'] = path_ct
            ds_ct_dict['Direction'] = image_ct.GetDirection()
            ds_ct_dict['Origin'] = image_ct.GetOrigin()
            ds_ct_dict['Size'] = image_ct.GetSize()
            ds_ct_dict['Spacing'] = image_ct.GetSpacing()

            # RTDOSE
            ds_rtdose_dict = dict()
            ds_rtdose_dict['ID'] = patient_id
            ds_rtdose_dict['Path_rtdose_file'] = path_rtdose_file
            ds_rtdose_dict['Path_rtdose'] = path_rtdose
            ds_rtdose_dict['Path_ct'] = path_ct
            ds_rtdose_dict['Dose_grid_scaling'] = float(ds.DoseGridScaling)
            ds_rtdose_dict['Dose_units'] = ds.DoseUnits

            #####
            logger.my_print('\tSTEP 4 (RTDOSE) - Resampling Image...')
            resampled_arr_rtdose, arr_ct, \
            arr_rtdose, resampled_image_rtdose = resample_rtdose_as_ct(image_rtdose, image_ct)

            # Make sure the two arrays are not identical
            assert not np.array_equal(resampled_arr_rtdose, arr_ct)

            # Convert to desired datatype
            # arr_ct = arr_ct.astype(cfg.ct_dtype)
            # Convert pixel value to DOSE value (cGy)
            arr_rtdose = (arr_rtdose * ds_rtdose_dict['Dose_grid_scaling'] * 100).astype(cfg.rtdose_dtype)
            resampled_arr_rtdose = (resampled_arr_rtdose * ds_rtdose_dict['Dose_grid_scaling'] * 100).astype(
                cfg.rtdose_dtype)

            #####
            logger.my_print('\tSTEP 5 (RTDOSE) - Extracting meta-data: before data processing and after data processing...')
            meta_data_list = ['Raw', 'Processed']
            image_rtdose_list = [image_rtdose, resampled_image_rtdose]
            arr_rtdose_list = [arr_rtdose, resampled_arr_rtdose]
            for key, im, arr in zip(meta_data_list, image_rtdose_list, arr_rtdose_list):
                ds_rtdose_dict_i = dict()
                ds_rtdose_dict_i['Direction'] = im.GetDirection()
                ds_rtdose_dict_i['Origin'] = im.GetOrigin()
                ds_rtdose_dict_i['Size'] = im.GetSize()
                ds_rtdose_dict_i['Spacing'] = im.GetSpacing()
                ds_rtdose_dict_i['Max_(cGy)'] = arr.max()
                ds_rtdose_dict_i['Min_(cGy)'] = arr.min()
                ds_rtdose_dict_i['Mean_(cGy)'] = arr.mean()
                ds_rtdose_dict[key] = ds_rtdose_dict_i

            #####
            logger.my_print('\tSTEP 6 (RTDOSE, CT) - Saving meta-data and arrays...')
            save_path_rtdose = os.path.join(save_dir_rtdose, patient_id)
            save_path_ct = os.path.join(save_dir_ct, patient_id)
            create_folder_if_not_exists(save_path_rtdose)
            create_folder_if_not_exists(save_path_ct)

            # Save meta-data dictionary as JSON
            # 'w': overwrites the file if the file exists. If the file does not exist, creates a new file for writing.
            # Source: https://tutorial.eyehunts.com/python/python-file-modes-open-write-append-r-r-w-w-x-etc/
            with open(os.path.join(save_path_rtdose, cfg.filename_rtdose_metadata_json), 'w') as file:
                json.dump(ds_rtdose_dict, file, default=set_default)

            with open(os.path.join(save_path_ct, cfg.filename_ct_metadata_json), 'w') as file:
                json.dump(ds_ct_dict, file, default=set_default)

            # Save as Numpy array
            np.save(file=os.path.join(save_path_ct, cfg.filename_ct_npy), arr=arr_ct)
            np.save(file=os.path.join(save_path_rtdose, cfg.filename_rtdose_npy), arr=resampled_arr_rtdose)

    end = time.time()
    logger.my_print('Elapsed time: {} seconds'.format(round(end - start, 3)))
    logger.my_print('DONE!')


if __name__ == '__main__':
    main()

