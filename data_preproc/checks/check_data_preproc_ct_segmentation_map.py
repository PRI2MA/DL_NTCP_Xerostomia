"""
main():
    - Get descriptive statistics of cropping regions.
"""
import os
import time
import pandas as pd

import data_preproc_config as cfg
from data_preproc_functions import Logger


def main():
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    logger = Logger(os.path.join(save_root_dir, cfg.filename_check_data_preproc_ct_segmentation_map_logging_txt))
    start = time.time()

    for _, d_i in zip(mode_list, data_dir_mode_list):
        # TODO: currently redundant
        # Load file from data_folder_types.py
        # TODO: temporary, for MDACC
        if use_umcg:
            df_data_folder_types = pd.read_csv(os.path.join(save_root_dir, cfg.filename_patient_folder_types_csv), sep=';',
                                               index_col=0)
            df_data_folder_types.index = ['%0.{}d'.format(patient_id_length) % int(x)
                                          for x in df_data_folder_types.index.values]

        # Load cropping regions of all patients
        cropping_regions = pd.read_csv(os.path.join(save_root_dir, cfg.filename_cropping_regions_csv),
                                       sep=';', index_col=0)

        # Get column_names for which we determine descriptive statistics
        column_names = cropping_regions.columns
        column_names = [x for x in column_names if x != 'patient_id']

        # Get descriptive statistics
        df = pd.DataFrame()
        for col_name in column_names:
            col = cropping_regions[col_name]
            col_min = col.min()
            col_mean = col.mean()
            col_median = col.median()
            col_max = col.max()
            df_i = pd.DataFrame([col_min, col_mean, col_median, col_max],
                                index=['min', 'mean', 'median', 'max'],
                                columns=[col_name])
            df = pd.concat([df, df_i], axis=1)

        df = df.round(cfg.nr_of_decimals)

        # Save descriptive statistics df as csv file
        df.to_csv(os.path.join(save_root_dir, cfg.filename_cropping_regions_stats_csv), sep=';')

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


if __name__ == '__main__':
    main()



