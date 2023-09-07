"""
Configuration file: containing global variables.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Whether to preprocess UMCG data or other data
use_umcg = True

# Paths
# TODO: temporary, for MDACC
if use_umcg:
    data_dir = '//zkh/appdata/RTDicom/PRI2MA/{}'
    data_collection_dir = '//zkh/appdata/RTDicom/HungChu/Data/PRI2MA'
    data_dir_citor = '//zkh/appdata/RTDicom/PRI2MA/{}'  # Contains all 1453 CITOR RTSTRUCTs
    save_dir_citor = 'D:/MyFirstData/raw_rtstructs_citor'  # Contains CITOR RTSTRUCTs for our patients in our cohort
    data_dir_dlc = '//zkh/appdata/RTDicom/PRI2MA/DLC'  # Contains (valid) DLC RTSTRUCTs
    save_root_dir = 'D:/MyFirstData'  # Contains saved files from scripts such as processed data, loggings, etc.
    save_root_dir_2 = '//zkh/appdata/RTDicom/HungChu/Data/PRI2MA'  # Contains saved files for dataset
    save_dir = os.path.join(save_root_dir_2, 'dicom_processed')
else:
    data_dir = '//zkh/appdata/RTDicom/PRI2MA/MDACC/{}'
    data_collection_dir = '//zkh/appdata/RTDicom/HungChu/Data/MDACC'
    data_dir_citor = '//zkh/appdata/RTDicom/PRI2MA/MDACC/{}'
    save_dir_citor = 'D:/MyFirstData/raw_rtstructs_mdacc'  # Contains CITOR RTSTRUCTs for our patients in our cohort
    data_dir_dlc = '//zkh/appdata/RTDicom/PRI2MA/MDACC/DLC UMCG'  # Contains (valid) DLC RTSTRUCTs
    save_root_dir = 'D:/MyFirstData_MDACC'  # Contains saved files from scripts such as processed data, loggings, etc.
    save_root_dir_2 = '//zkh/appdata/RTDicom/HungChu/Data/MDACC'  # Contains saved files for dataset
    save_dir = os.path.join(save_root_dir_2, 'dicom_processed')

# data_preproc_rtdose.py
save_dir_ct = save_dir
save_dir_rtdose = save_dir

# TODO: temporary, for MDACC
if use_umcg:
    # data_preproc_segmentation_map_citor.py
    save_dir_segmentation_map_citor = os.path.join(save_root_dir, 'segmentation_map_citor')  # CITOR segmentation_map
    # data_preproc_ct_segmentation_map_dlc.py
    save_dir_segmentation_map_dlc = os.path.join(save_root_dir, 'segmentation_map_dlc')  # DLC segmentation_map
    # data_preproc_ct_segmentation_map.py
    save_dir_segmentation_map = save_dir  # Every segmentation_map (CITOR + DLC)
    # data_preproc.py
    save_dir_dataset_full = os.path.join(save_root_dir, 'dataset_full')
    # save_dir_dataset_full = '//zkh/appdata/RTDicom/PRI2MA/preprocessed/dataset_full'
    save_dir_dataset = os.path.join(save_root_dir, 'dataset')
    # check_data_preproc.py
    save_dir_figures = os.path.join(save_root_dir, 'figures')
    save_dir_figures_all = os.path.join(save_dir_figures, 'all')
else:
    # data_preproc_segmentation_map_citor.py
    save_dir_segmentation_map_citor = os.path.join(save_root_dir, 'segmentation_map_mdacc')  # CITOR segmentation_map
    save_dir_segmentation_map_dlc = os.path.join(save_root_dir, 'segmentation_map_dlc_mdacc')  # DLC segmentation_map
    save_dir_segmentation_map = save_dir
    save_dir_dataset_full = os.path.join(save_root_dir, 'dataset_full_mdacc')
    # save_dir_dataset_full = '//zkh/appdata/RTDicom/PRI2MA/MDACC/preprocessed/dataset_full_mdacc'
    save_dir_dataset = os.path.join(save_root_dir, 'dataset_mdacc')
    save_dir_figures = os.path.join(save_root_dir, 'figures_mdacc')
    save_dir_figures_all = os.path.join(save_dir_figures, 'all')


# data_folder_types.py
# Available patient's folder types (e.g. 'with_contrast'/'no_constrast')
patient_folder_types = ['with_contrast', 'no_contrast']
# check_folder_structure.py
# TODO: temporary, for MDACC
# File extensions: required for finding the files. Usually this is '.dcm', but MDACC has no file extension, so ''.
# Note: these variables are only used in check_folder_structure.py, because in the remainder of the code
# we can assume that the folder structure is correct and hence glob('/*') and glob('/*.dcm') would output the same.
if use_umcg:
    ct_file_ext = '.dcm'  # '.dcm' (UMCG) | '' (MDACC)
    rtdose_file_ext = '.dcm'
    rtstruct_file_ext = '.dcm'
else:
    ct_file_ext = ''  # '.dcm' (UMCG) | '' (MDACC)
    rtdose_file_ext = ''
    rtstruct_file_ext = ''

# Filenames
# data_folder_types.py
filename_patient_folder_types_csv = 'patient_folder_types.csv'
# check_rtstruct.py
filename_overview_all_structures_csv = 'overview_all_structures.csv'
# data_preproc_ct_rtdose.py
filename_rtdose_dcm = 'Echte_Original_plan.dcm'
# check_folder_structure.py
filename_check_folder_structure_logging_txt = 'check_folder_structure_logging.txt'
filename_patients_incorrect_folder_structure_csv = 'patients_incorrect_folder_structure_{}.csv'
# check_data.py
filename_structures_values_json = 'structures_values.json'
filename_check_data_logging_txt = 'check_data_logging.txt'
filename_patients_in_citor_and_dlc_txt = 'patients_in_citor_and_dlc.txt'
# folders.py
filename_frame_of_reference_uid_csv = 'frame_of_reference_uid.csv'
filename_invalid_uid_csv = 'invalid_uid.csv'
# data_preproc_ct_rtdose.py
filename_data_preproc_ct_rtdose_logging_txt = 'data_preproc_ct_rtdose_logging.txt'
filename_ct_npy = 'ct.npy'  # CT (Numpy array)
filename_rtdose_npy = 'rtdose.npy'  # RTDOSE (Numpy array), has the same shape as ct.npy
filename_rtdose_metadata_json = 'rtdose_metadata.json'  # meta-data of the RTDOSE data
filename_ct_metadata_json = 'ct_metadata.json'  # meta-data of the CT data
# check_data_preproc_ct_rtdose.py
filename_check_data_preproc_ct_rtdose_logging_txt = 'check_data_preproc_ct_rtdose_logging.txt'
filename_ct_npy = 'ct.npy'
filename_rtdose_npy = 'rtdose.npy'
filename_overview_ct_metadata_csv = 'overview_ct_metadata.csv'
filename_overview_ct_csv = 'overview_ct.csv'
filename_overview_rtdose_csv = 'overview_rtdose.csv'
# data_preproc_segmentation_map_citor.py
filename_data_preproc_ct_segmentation_map_citor_logging_txt = 'data_preproc_ct_segmentation_map_citor_logging.txt'
filename_data_preproc_ct_segmentation_map_best_logging_txt = 'data_preproc_ct_segmentation_map_best_logging.txt'
filename_structures_with_contouring_issues_citor_i_txt = 'structures_with_contouring_issues_citor_{i}.txt'
filename_patients_with_contouring_issues_citor_txt = 'patients_with_contouring_issues_citor.txt'
filename_patient_ids_with_filename_csv = 'patient_ids_with_filename.csv'
filename_overview_all_structures_count_citor_csv = 'overview_all_structures_count_citor.csv'
filename_overview_structures_count_citor_csv = 'overview_structures_count_citor.csv'
# data_preproc_ct_segmentation_map_dlc.py
filename_data_preproc_ct_segmentation_map_dlc_logging_txt = 'data_preproc_ct_segmentation_map_dlc_logging.txt'
# data_preproc_ct_segmentation_map.py
filename_data_preproc_ct_segmentation_map_logging_txt = 'data_preproc_ct_segmentation_map_logging.txt'
filename_data_preproc_ct_segmentation_map_cropping_logging_txt = 'data_preproc_ct_segmentation_map_cropping_logging.txt'
filename_segmentation_map_npy = 'segmentation_map.npy'
filename_segmentation_map_i_npy = 'segmentation_map_{i}.npy'  # segmentation_map of all structures
filename_structures_value_count_json = 'structures_value_count.json'
filename_structures_value_count_i_json = 'structures_value_count_{i}.json'  # for each relevant structure: contains
# segmentation label map value and the number of voxel data points (count)
filename_overview_structures_count_dlc_csv = 'overview_structures_count_dlc.csv'
# most voxel data points + corresponding count
filename_structures_with_contouring_issues_dlc_txt = 'structures_with_contouring_issues_dlc.txt'
# filename_structures_with_contouring_issues_i_txt = 'structures_with_contouring_issues_{i}.txt'
filename_patients_with_contouring_issues_dlc_txt = 'patients_with_contouring_issues_dlc.txt'
# filename_patients_with_contouring_issues_txt = 'patients_with_contouring_issues.txt'
filename_best_count_files_json = 'best_count_files.json'  # for each relevant structure: filename that contains
# most voxel data points + the corresponding count
filename_cropping_regions_csv = 'cropping_regions.csv'  # for each patient and each dimension: contains the upper and
# lower cropping coordinate and the length (i.e. distance between upper and lower coordinate)
# count >= minimal_count
# check_data_preproc_ct_segmentation_map.py
filename_check_data_preproc_ct_segmentation_map_logging_txt = 'check_data_preproc_ct_segmentation_map_logging.txt'
filename_cropping_regions_stats_csv = 'cropping_regions_stats.csv'
filename_segmentation_fraction_outside_ct_csv = 'segmentation_fraction_outside_ct.csv'
# filename_all_structures_csv = 'all_structures.csv'  # all structures of all patients
# filename_valid_and_relevant_structures_csv = 'valid_and_relevant_structures.csv'  # valid structure if
# data_preproc.py
filename_data_preproc_array_logging_txt = 'data_preproc_array_logging.txt'
filename_data_preproc_features_logging_txt = 'data_preproc_features_logging.txt'
filename_endpoints_csv = 'endpoints.csv'
filename_patient_id_npy = '{patient_id}.npy'  # model's input (Numpy array), concatenation of: CT, RTDOSE and RTSTRUCT
filename_features_csv = 'features.csv'
filename_stratified_sampling_csv = 'stratified_sampling.csv'
# segmentation arrays
filename_overview_structures_count_csv = 'overview_structures_count.csv'
# check_data_preproc.py
filename_check_data_preproc_logging_txt = 'check_data_preproc_logging.txt'
# data_preproc_exclude_patient.py
if use_umcg:
    filename_exclude_patients_csv = 'exclude_patients_umcg.csv'
else:
    filename_exclude_patients_csv = 'exclude_patients_mdacc.csv'
# main.py
filename_main_logging_txt = 'main_logging.txt'

# Modelling
# data_preproc.py: filename_endpoints_csv
patient_id_col = 'PatientID'
endpoint = 'HN35_Xerostomia_M12_class'
baseline_col = 'HN35_Xerostomia_W01_class'
submodels_features = [
    ['Submandibular_meandose', 'HN35_Xerostomia_W01_little', 'HN35_Xerostomia_W01_moderate_to_severe'],
    ['Parotid_meandose_adj', 'HN35_Xerostomia_W01_little', 'HN35_Xerostomia_W01_moderate_to_severe'],
]  # Features of submodels. Should be a list of lists. len(submodels_features) = nr_of_submodels. If None, then
# no fitting of submodels.
features = ['Submandibular_meandose', 'Parotid_meandose_adj', 'HN35_Xerostomia_W01_little',
            'HN35_Xerostomia_W01_moderate_to_severe']  # Features of final model. Elements in submodels_features should
# be a subset of `features`, i.e. submodels_features can have more distinct features than the final model.
lr_coefficients = None  # [-2.9032, 0.0193, 0.1054, 0.5234, 1.2763]  # Values starting with coefficient for `intercept`,
# followed by coefficients of `features` (in the same order). If None, then no predefined coefficients will be used.
ext_features = ['HN35_Xerostomia_W01_not_at_all', 'CT+C_available', 'CT_Artefact', 'Photons', 'Loctum2_v2', 'Split',
                'Gender', 'Age']
# (Stratified Sampling)
# data_preproc.py, check_data_preproc_ct_segmentation_map.py: relevant segmentation structures
parotis_structures = ['parotis_li', 'parotis_re']
submand_structures = ['submandibularis_li', 'submandibularis_re']
lower_structures = ['crico', 'thyroid']
mandible_structures = ['mandible']
other_structures = ['glotticarea', 'oralcavity_ext', 'supraglottic', 'buccalmucosa_li',
                    'buccalmucosa_re', 'pcm_inf', 'pcm_med', 'pcm_sup', 'esophagus_cerv']
structures_compound_list = [parotis_structures, submand_structures, lower_structures, mandible_structures,
                            other_structures]
structures_uncompound_list = [i for x in structures_compound_list for i in x]
# E.g. structures_uncompound_list: ['parotid_li', 'parotid_re', 'submandibular_li', ...]
structures_values = dict(zip(structures_uncompound_list,
                             range(1, len(structures_uncompound_list) + 1)))
# E.g.: structures_values: {'parotid_li': 1, 'parotid_re': 2, 'submandibular_li': 3, ...}
max_outside_fraction = 0.1  # maximum fraction that the segmentation_mask of a structure is allowed to be outside CT
max_outside_lower_limit = -200  # lower limit CT intensity (HU) for which segmentation_mask should correspond to

# Data collection/preparation
# data_collection.py
filename_patient_ids = 'Patient_ids_{}.csv'
# TODO: important note: currently only len(mode_list) == len(save_dir_mode_list) == 1 is allowed!
# TODO: a possibility to extend is to add '/{}'.format(d_i) to save_root_dir
# TODO: temporary, for MDACC
if use_umcg:
    mode_list = ['now']  # 'now', 'later', 'imputation', 'new_patients']
    save_dir_mode_list = [
     'Now_Has_endpoint_for_at_least_1_toxicity']  # 'Now_Has_endpoint_for_at_least_1_toxicity', 'Later - Could have endpoint later', 'Patients for imputation', 'to_be_processed_by_hung_do_not_change_folder']
else:
    mode_list = ['mdacc']
    save_dir_mode_list = ['DICOM_DATA_anonymized_cleaned']

# mode_list = ['new_patients']  # , 'later', 'imputation']
# save_dir_mode_list = ['New_patients']  # , 'Later - Could have endpoint later', 'Patients for imputation']
# save_dir_mode_list = ['Postoperative patients/Now - Has endpoint for at least 1 toxicity',
#                       'Postoperative patients/Later - Could have endpoint later',
#                       'Postoperative patients/Patients for imputation']
# folders.py
# study_uid_tag = '0020000D'  # Study Instance UID
# series_uid_tag = '0020000E'  # Series Instance UID
frame_of_ref_uid_tag = '00200052'
patient_id_tag = '00100020'
# data_preproc_ct_rtdose.py/data_preproc_ct_segmentation_map.py
ct_dtype = 'int16'
rtdose_dtype = 'uint16'
# segmentation_dtype = 'int16'
# data_preproc_ct_segmentation_map_citor.py/data_preproc_ct_segmentation_map.py
keep_segmentation_map_elements = True  # In case of adding segmentation maps with overlapping elements: whether to keep
# overlapping elements from existing segmentation map (True) or the new segmentation map (False)
# data_preproc_ct_segmentation_map.py
minimal_count = 100  # Count lower bound: for each patient: if the number of voxels for a structure is more than
# or equal to this value, then the segmentation_map of the structure will be considered to be valid
nr_of_decimals = 2  # Number of decimals used for saving in (csv) files
# data_preproc.py, data_preproc_ct_segmentation_map.py
# Convert filtered Pandas column to list, and convert patient_id = 111766 (type=int, because of Excel file) to
# patient_id = '0111766' (type=str)
unfill_holes = True  # Whether or not to consider unfilled holes of contours (see oralcavity_ext structure)
# TODO: temporary, for MDACC
if use_umcg:
    patient_id_length = 7
else:
    patient_id_length = 10
perform_spacing_correction = True
perform_cropping = True
perform_clipping = True
perform_transformation = False
spacing = [2., 2., 2.]  # (only if perform_spacing=True) desired voxel spacing
y_center_shift_perc = 0.0  # (only if perform_cropping=True) percentage of total num_rows to shift the y_center. If
# y_center_shift_perc > 0, then the bounding box will be placed on higher y-indices of the array. Vice versa for
# y_center_shift_perc < 0. If y_center_shift_perc = 0, then y_center will not be shifted.
bb_size = [100, 100, 100]  # (only if perform_cropping=True) bounding box (/cropping) with
# shape = [z_size, y_size, x_size] = [num_slices, num_rows, num_columns] (default of Numpy) given in the spacing (!!!)
input_size = [128, 128, 128]  # (only if perform_transformation=True) model input size,
# shape = [z, y, x] = [num_slices, num_rows, num_columns] (default of Numpy)
label_resize_mode = 'nearest'  # (only if perform_transformation=True)
non_label_resize_mode = 'trilinear'  # (only if perform_transformation=True)
ct_min = -1000  # (only if perform_clipping=True) CT pixel value lower bound (Hounsfield Units)
ct_max = 1000  # (only if perform_clipping=True) CT pixel value upper bound (Hounsfield Units)
# data_preproc_exclude_patient.py
rtdose_lower_limit = 6000  # Minimum rtdose.max() allowed
rtdose_upper_limit = 8500  # Maximum rtdose.max() allowed

# Plotting
# check_data_preproc.py
ct_ww = 600  # (plotting) Window Width: the range of the numbers (i.e. max - min)
ct_wl = 100  # (plotting) Window Level: the center of the range
plot_nr_images = 36  # If None, then consider all slices
figsize = (12, 12)
plot_nr_images_multiple = 8
figsize_multiple = (18, 9)
# Colormap
ct_cmap = 'gray'
# Custom cmap for RTDOSE: sample from the colormap 'jet', consider 256 colors in total.
# The upper part (0.8 - 0.9) (orange --> red) of plt.cm.jet will be mapped to upper int(256 * 0.05) = 12 values
# out of 256 values
# The middle part (green --> orange) of plt.cm.jet will be mapped to mid 256 - 12 - 201 = 43 values out of 256 values
# The lower half (blue --> green) of plt.cm.jet will be mapped to lower 201 values out of 256 values
# Result: map `blue (0) --> green (3325) --> red (6650)` to `blue (0) --> green (5225) --> orange () --> red (6650)`
# Source: https://stackoverflow.com/questions/49367144/modify-matplotlib-colormap
rtdose_vmax = 7000 * 0.95  # = 6650
rtdose_vmid = 5500 * 0.95  # = 5225
rtdose_vmin = 0  # optional (else: None)
# 0.9: jet colormap at 90% is red (whereas 100% is dark red)
rtdose_cmap_upper = plt.cm.jet(np.linspace(0.8, 0.9, int(256 * 0.05)))
rtdose_cmap_mid = plt.cm.jet(np.linspace(0.5, 0.8, 256 - int(256 * 0.05) - int(rtdose_vmid / rtdose_vmax * 256)))
rtdose_cmap_lower = plt.cm.jet(np.linspace(0, 0.5, int(rtdose_vmid / rtdose_vmax * 256)))  # int(5225/6650 * 256) = 201
rtdose_cmap = colors.LinearSegmentedColormap.from_list('rtdose_cmap', np.vstack((rtdose_cmap_lower, rtdose_cmap_mid,
                                                                                 rtdose_cmap_upper)))
segmentation_cmap = 'gray'
draw_contour_structures = parotis_structures + submand_structures  # create contour of these structures in the figures
draw_contour_color = 'r'  # color of the contour line
draw_contour_linewidth = 0.25  # width of the contour line
ct_ticks_steps = 100  # optional (else: None)
rtdose_ticks_steps = 1000  # optional (else: None)
segmentation_ticks_steps = 1  # optional (else: None)
ct_colorbar_title = 'HU'
rtdose_colorbar_title = 'cGy'
segmentation_colorbar_title = ''

# Run whole data_preproc pipeline for a small number of patients, useful for testing
test_patients_list = None  # ['0276627']
