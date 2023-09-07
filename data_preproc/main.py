import os
import time

import data_preproc_config as cfg
from data_preproc_functions import Logger

import data_folder_types
import data_collection
import data_preproc_ct_rtdose
import data_preproc_ct_segmentation_map_citor
import data_preproc_ct_segmentation_map_dlc
import data_preproc_ct_segmentation_map
import data_preproc
import data_preproc_exclude_patients
import checks.check_folder_structure as check_folder_structure
import checks.check_rtstruct as check_rtstruct
import checks.check_rtdose as check_rtdose
import checks.check_data as check_data
import checks.check_data_preproc_ct_rtdose as check_data_preproc_ct_rtdose
import checks.check_data_preproc_ct_segmentation_map as check_data_preproc_ct_segmentation_map
import checks.check_data_preproc as check_data_preproc

if __name__ == '__main__':
    # Initialize variables
    save_root_dir = cfg.save_root_dir
    logger = Logger(os.path.join(save_root_dir, cfg.filename_main_logging_txt))
    start = time.time()

    # Run the separate modules
    logger.my_print('Running data_folder_types...')
    data_folder_types.main()

    logger.my_print('Running check_folder_structure...')
    check_folder_structure.main()

    logger.my_print('Running data_collection...')
    data_collection.main()


    # # The following checks can be skipped
    # # logger.my_print('Running check_rtstruct...')
    # # check_rtstruct.main()  # may be skipped
    # #
    # # logger.my_print('Running check_rtdose...')
    # # check_rtdose.main()  # can be skipped
    # #
    # # logger.my_print('Running check_data...')
    # # check_data.main()  # can be skipped


    # logger.my_print('Running data_preproc_ct_rtdose...')
    # data_preproc_ct_rtdose.main()

    # logger.my_print('Running check_data_preproc_ct_rtdose...')
    # check_data_preproc_ct_rtdose.main()

    # logger.my_print('Running data_preproc_segmentation_map_citor...')
    # data_preproc_ct_segmentation_map_citor.main()
    #
    # logger.my_print('Running data_preproc_ct_segmentation_map_dlc...')
    # data_preproc_ct_segmentation_map_dlc.main()
    #
    # logger.my_print('Running data_preproc_ct_segmentation_map...')
    # data_preproc_ct_segmentation_map.main()
    #
    # logger.my_print('Running check_data_preproc_ct_segmentation_map...')
    # check_data_preproc_ct_segmentation_map.main()

    # logger.my_print('Running data_preproc...')
    # data_preproc.main()

    logger.my_print('Running check_data_preproc...')
    check_data_preproc.main()
    #
    # logger.my_print('Running data_preproc_exclude_patients...')
    # # Note: if len(test_patients_list) > 0, but we need exclude_patients.csv for all patients, then Search for:
    # # `# TODO: temporary: consider all patients, for exclude_patients.csv`, and uncomment/enable the code lines.
    # data_preproc_exclude_patients.main()

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')
