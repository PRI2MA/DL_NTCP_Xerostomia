"""
This script:
    - Creates and saves plot for the outputs arrays of data_preproc.py.
    - For each patient and each structure: for segmentation_map: calculate percentage of voxels that are at CT's voxel,
        where CTs voxel values are less than or equal to some value (e.g. -200 HU). This is to check which structure
        lies outside the actual human body. This could indicate incorrect segmentation.

Useful note:
    - plt.imshow(): the dimensions (M, N) of input array define the rows and columns of the image, respectively.
        I.e. in the figures of an array with shape (z, y, x) = (z, M, N) = (num_slices, num_rows, num_columns):
        - z-axis: scan slice. Starting with slice 0 (i.e. bottom human body part in the scan), until last slice
            (upper human body part in the scan). Subplots in the figure are created from left to right, top to bottom.
        - y-axis (rows): moves up and down of the image.
        - x-axis (columns): moves left and right of the image.
    - main_segmentation_outside_ct() with "ct < max_outside_lower_limit" does not work for slices with metal artifacts.
        That is, the metal artifacts changes the HU values of inside CT. As a result  many voxels will be considered
        "outside" the CT, but are actually inside the CT.

    Source: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    - The very first and very last slices are always included in the subplot.
    - The pixel values of the RTDOSE subplots cannot be compared between RTDOSE subplots, because the subplots are
        created individually and so the pixel values are normalized differently for each subplot.
"""
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# Source: https://github.com/matplotlib/mplfinance/issues/386
import matplotlib.pyplot as plt
from tqdm import tqdm

import data_preproc_config as cfg
from data_preproc_functions import create_folder_if_not_exists, Logger, sort_human


def plot_arr(arr, nr_images, figsize, cmap, colorbar_title, filename, vmin=None, vmax=None, ticks_steps=None,
             segmentation_map=None):
    """
    Plot and save slices of array. The slices that we consider are equal distant from each other in the whole array.
    For example, if the array has 100 slices, and nr_images = 4, then the slices with the following index
    will be plotted and saved: [0, 33, 66, 99].
    The first and last plot will always be the first and last slice, respectively.

    If nr_images is None, then consider all slices!

    Result of determining the number of columns and rows:
        nr_slices: 1
            num_cols: 1
            num_rows: 1
        nr_slices: 2
            num_cols: 1
            num_rows: 2
        nr_slices: 3
            num_cols: 2
            num_rows: 2
        nr_slices: 4
            num_cols: 2
            num_rows: 2
        nr_slices: 5
            num_cols: 2
            num_rows: 3
        ...

    Args:
        arr (Numpy array): array with slices to be plotted, of shape (z, y, x) = (num_slices, num_rows, num_columns)
        nr_images (int): number of slices to plot
        figsize (tuple): size of output figure
        cmap (str): colormap, see https://matplotlib.org/stable/tutorials/colors/colormaps.html
        colorbar_title (str): title of colorbar
        filename (str): save location
        vmin (int/float): (optional) minimum value for colormap: smaller values will be clipped to this minimum value
        vmax (int/float): (optional) maximum value for colormap: larger values will be clipped to this maximum value
        ticks_steps (int): (optional) steps of value display for colormap
        segmentation_map (Numpy array): (optional) segmentation_map for drawing the contours of the structures

    Returns:

    """
    # Initialize variables
    nr_slices = arr.shape[0]
    if nr_images is None:
        nr_images = nr_slices
    # List of structures for which we may plot contours (i.e. if segmentation_map is not None)
    structures = cfg.draw_contour_structures

    # Make sure that nr_images that we want to plot is greater than or equal to the number of slices available
    if nr_slices < nr_images:
        nr_images = nr_slices
    slice_indices = np.linspace(0, nr_slices - 1, num=nr_images)

    # Only consider unique values
    slice_indices = np.unique(slice_indices.astype(int))

    # Estimate the number of columns and rows
    num_cols_rows = len(slice_indices) ** (1 / 2)
    # Determine first decimal of square-root: this is important in determining the exact number of columns and rows
    # num_cols_rows = 5.50001 results in num_cols_rows_without_rounding = 5.5
    # num_cols_rows = 5.49999 results in num_cols_rows_without_rounding = 5.4
    num_cols_rows_without_rounding = math.floor(num_cols_rows * 10) / 10
    base_num_cols_rows = int(num_cols_rows_without_rounding)
    decimal_num_cols_rows = num_cols_rows_without_rounding - base_num_cols_rows
    if decimal_num_cols_rows == 0:
        num_cols = num_rows = base_num_cols_rows
    elif decimal_num_cols_rows < 0.5:
        num_cols = base_num_cols_rows
        num_rows = base_num_cols_rows + 1
    else:
        num_cols = base_num_cols_rows + 1
        num_rows = base_num_cols_rows + 1

    # # Converting to log()-values is useful for large value differences, such as in RTDOSE
    # # Map arr from [a, b] to [1, b-a+1] (e.g. from [-1000, 1000] to [1, 2001]) before applying log()
    # arr = (arr - arr.min()) + 1
    # arr = np.log(arr)

    # Colormap
    # Data range
    vmin = vmin if vmin is not None else arr.min()
    vmax = vmax if vmax is not None else arr.max()
    # Source: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    # Label ticks
    # '+1' because end of interval is not included
    ticks = np.arange(vmin, vmax+1, ticks_steps) if ticks_steps is not None else None

    # fig = plt.figure(figsize=tuple(figsize))
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=tuple(figsize))
    ax = ax.flatten()

    for i, idx in enumerate(slice_indices):
        # Consider the first and last slice
        if i == 0:
            idx = 0
        if i == nr_images - 1:
            idx = nr_slices - 1

        idx = int(idx)
        im = ax[i].imshow(arr[idx, ...], cmap=cmap, vmin=vmin, vmax=vmax)
        if segmentation_map is not None:
            for struct in structures:
                struct_value = cfg.structures_values[struct]
                ax[i].contour(segmentation_map[idx, ...] == struct_value,
                              colors=cfg.draw_contour_color,
                              linewidths=cfg.draw_contour_linewidth)
        ax[i].axis('off')

    plt.tight_layout()

    # Add colorbar
    fig.subplots_adjust(right=0.8)
    # add_axes([xmin, ymin, dx, dy])
    cbar = fig.add_axes([0.825, 0.15, 0.02, 0.7])
    cbar.set_title(colorbar_title)
    fig.colorbar(im, cax=cbar, ticks=ticks)

    plt.savefig(filename)
    plt.close(fig)
    # Source: https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib


def plot_multiple_arrs(arr_list, overlay_list, alpha_overlay_list, nr_images, figsize, cmap_list, cmap_overlay_list,
                       colorbar_title_list, filename, vmin_list, vmin_overlay_list, vmax_list, vmax_overlay_list,
                       ticks_steps_list, segmentation_map_list):
    """
    Plot slices of multiple arrays. Each Numpy on a different row, e.g. CT (row 1), RTDOSE (row 2) and
    segmentation_map (row 3).

    Args:
        arr_list (list): list of Numpy arrays with slices to be plotted, each Numpy array with shape
            (z, y, x) = (num_slices, num_rows, num_columns)
        overlay_list (list): list of Numpy arrays with slices to be plotted overlayed on arr_list.
            Note: len(arr_list) == len(overlay_list). If overlay_list[i] = None, then no overlay.
        alpha_overlay_list (float): overlay alpha, in [0, 1].
        nr_images (int): number of slices to plot for each input
        figsize (tuple): size of output figure
        cmap_list (list): list of cmaps
        cmap_overlay_list (list): list of cmaps for arrays in overlay_list
        colorbar_title_list (list): list of titles of colorbar
        filename (str): save location
        vmin_list (list): list minimum value for colormap
        vmin_overlay_list (list): list minimum value for arrays in overlay_list
        vmax_list (list): list minimum value for colormap
        vmax_overlay_list (list): list minimum value for colormap for arrays in overlay_list
        ticks_steps_list (list): list of steps of value display for colormap
        segmentation_map_list (list): list of segmentation_maps for drawing the contours of the structures

    Returns:

    """
    assert len(arr_list) == len(overlay_list)

    # Make sure that every input array has the same number of slices
    nr_slices = arr_list[0].shape[0]
    for i in range(1, len(arr_list)):
        assert nr_slices == arr_list[i].shape[0]
    # List of structures for which we may plot contours (i.e. if segmentation_map is not None)
    structures = cfg.draw_contour_structures

    # Initialize variables
    if nr_images is None:
        nr_images = nr_slices

    # Make sure that nr_images that we want to plot is greater than or equal to the number of slices available
    if nr_slices < nr_images:
        nr_images = nr_slices
    slice_indices = np.linspace(0, nr_slices - 1, num=nr_images)

    # Only consider unique values
    slice_indices = np.unique(slice_indices.astype(int))

    # Determine number of columns and rows
    num_cols = nr_images
    num_rows = len(arr_list)

    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=tuple(figsize))

    for row, (arr, arr_overlay) in enumerate(zip(arr_list, overlay_list)):
        # Make sure that arr and arr_overlay shapes are identical
        if arr_overlay is not None:
            assert arr.shape == arr_overlay.shape
            alpha_overlay = alpha_overlay_list[row]
            cmap_overlay = cmap_overlay_list[row]
            vmin_overlay = vmin_overlay_list[row] if vmin_overlay_list[row] is not None else arr_overlay.min()
            vmax_overlay = vmax_overlay_list[row] if vmax_overlay_list[row] is not None else arr_overlay.max()

        cmap = cmap_list[row]
        colorbar_title = colorbar_title_list[row]

        # Colormap
        # Data range
        vmin = vmin_list[row] if vmin_list[row] is not None else arr.min()
        vmax = vmax_list[row] if vmax_list[row] is not None else arr.max()

        # Label ticks
        # '+1' because end of interval is not included
        ticks = np.arange(vmin, vmax + 1, ticks_steps_list[row]) if ticks_steps_list[row] is not None else None

        for i, idx in enumerate(slice_indices):
            # Consider the first and last slice
            if i == 0:
                idx = 0
            if i == nr_images - 1:
                idx = nr_slices - 1

            idx = int(idx)
            im = ax[row, i].imshow(arr[idx, ...], cmap=cmap, vmin=vmin, vmax=vmax)

            if arr_overlay is not None:
                im = ax[row, i].imshow(arr_overlay[idx, ...], alpha=alpha_overlay, cmap=cmap_overlay, vmin=vmin_overlay,
                                       vmax=vmax_overlay)

            if segmentation_map_list[row] is not None:
                for struct in structures:
                    struct_value = cfg.structures_values[struct]
                    ax[row, i].contour(segmentation_map_list[row][idx, ...] == struct_value,
                                       colors=cfg.draw_contour_color,
                                       linewidths=cfg.draw_contour_linewidth)
            ax[row, i].axis('off')

        plt.tight_layout()

        # Add colorbar
        if colorbar_title is not None:
            fig.subplots_adjust(right=0.8)
            max_height = 0.925
            min_height = 1 - max_height
            length = max_height - min_height
            length_per_input = length / num_rows
            epsilon = 0.05
            bottom = max_height - (row + 1) * length_per_input + epsilon / 2
            cbar = fig.add_axes(rect=[0.825, bottom, 0.01, length_per_input - epsilon])
            cbar.set_title(colorbar_title)
            fig.colorbar(im, cax=cbar, ticks=ticks)

    plt.savefig(filename)
    plt.close(fig)


def main_segmentation_outside_ct():
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    save_dir_dataset_full = cfg.save_dir_dataset_full
    max_outside_fraction = cfg.max_outside_fraction
    max_outside_lower_limit = cfg.max_outside_lower_limit
    logger = Logger(os.path.join(save_root_dir, cfg.filename_data_preproc_ct_segmentation_map_logging_txt))
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

        # Get patient_ids
        patients_list = os.listdir(os.path.join(save_dir_dataset_full))
        patients_list = sort_human(patients_list)
        n_patients = len(patients_list)
        logger.my_print('Total number of patients: {n}'.format(n=n_patients))

        # TODO: temporary: consider all patients, for exclude_patients.csv
        # Testing for a small number of patients
        if cfg.test_patients_list is not None:
            test_patients_list = cfg.test_patients_list
            patients_list = [x for x in test_patients_list if x in patients_list]

        # Make sure that the order and length is equal
        # patients_list = [x.replace('.npy', '') for x in patients_npy_list]
        # assert patients_npy_list == [x + '.npy' for x in patients_list]

        df = pd.DataFrame()
        for patient_id in tqdm(patients_list):
            logger.my_print('Patient_id: {id}'.format(id=patient_id))

            # Determine folder type of interest ('with_contrast'/'no_contrast')
            # TODO: temporary, for MDACC
            if use_umcg:
                folder_type_i = df_data_folder_types.loc[patient_id]['Folder']
            else:
                folder_type_i = ''

            # Load CT and segmentation
            # arr = np.load(os.path.join(save_dir_dataset_full, patient_id_npy))
            ct = np.load(os.path.join(save_dir_dataset_full, patient_id, cfg.filename_ct_npy))
            segmentation_map = np.load(os.path.join(save_dir_dataset_full, patient_id, cfg.filename_segmentation_map_npy))

            assert ct.shape == segmentation_map.shape

            outside_dict = dict()
            for structure in cfg.parotis_structures + cfg.submand_structures:
                # Total segmentation_map
                mask = (segmentation_map == cfg.structures_values[structure])
                total_nr_voxels = mask.sum()

                # Segmentation_map outside CT of human body
                mask_outside = (ct < max_outside_lower_limit) * mask
                outside_nr_voxels = mask_outside.sum()

                # Determine fraction of segmentation outside CT of human body
                fraction_outside = outside_nr_voxels / total_nr_voxels
                if fraction_outside > max_outside_fraction:
                    logger.my_print('Patient_id = {}, structure = {}, fraction outside CT = {}'
                                    .format(patient_id, structure, fraction_outside),
                                    level='warning')
                outside_dict[structure] = fraction_outside

            # Save descriptive statistics
            df_i = pd.DataFrame(outside_dict, index=[patient_id])
            df = pd.concat([df, df_i], axis=0)

        # Save as csv file
        df = df.round(cfg.nr_of_decimals)
        df.to_csv(os.path.join(save_root_dir, cfg.filename_segmentation_fraction_outside_ct_csv), sep=';')

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main_plot():
    # Initialize variables
    use_umcg = cfg.use_umcg
    mode_list = cfg.mode_list
    data_dir_mode_list = cfg.save_dir_mode_list
    patient_id_length = cfg.patient_id_length

    save_root_dir = cfg.save_root_dir
    save_dir_dataset_full = cfg.save_dir_dataset_full
    figures_dir = cfg.save_dir_figures
    figures_all_dir = cfg.save_dir_figures_all
    logger = Logger(os.path.join(save_root_dir, cfg.filename_check_data_preproc_logging_txt))
    figsize = cfg.figsize
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

        # Create folder if not exist
        create_folder_if_not_exists(figures_dir)
        create_folder_if_not_exists(figures_all_dir)

        # Get patient_ids
        patients_list = os.listdir(os.path.join(save_dir_dataset_full))
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

            # Create folder if not exist
            create_folder_if_not_exists(os.path.join(figures_dir, patient_id))

            # Load npy data of shape (1, z, y, x)
            ct_arr = np.load(os.path.join(save_dir_dataset_full, patient_id, cfg.filename_ct_npy))[0]
            rtdose_arr = np.load(os.path.join(save_dir_dataset_full, patient_id, cfg.filename_rtdose_npy))[0]
            segmentation_map_arr = np.load(os.path.join(save_dir_dataset_full, patient_id,
                                                        cfg.filename_segmentation_map_npy))[0]

            # CT (with Windowing)
            # ct_arr = arr[0]
            ct_min_plot = cfg.ct_wl - cfg.ct_ww / 2
            ct_max_plot = cfg.ct_wl + cfg.ct_ww / 2
            # Source: https://myctregistryreview.com/courses/my-ct-registry-review-demo/lessons/ct-physics/topic/window-width-and-window-level/
            ct_arr = np.clip(ct_arr, a_min=ct_min_plot, a_max=ct_max_plot)
            plot_arr(ct_arr, nr_images=cfg.plot_nr_images, figsize=figsize, cmap=cfg.ct_cmap,
                     colorbar_title=cfg.ct_colorbar_title,
                     filename=os.path.join(figures_dir, patient_id, 'ct.png'), ticks_steps=cfg.ct_ticks_steps,
                     segmentation_map=segmentation_map_arr)

            # RTDOSE
            # rtdose_arr = arr[1]
            plot_arr(rtdose_arr, nr_images=cfg.plot_nr_images, figsize=figsize, cmap=cfg.rtdose_cmap,
                     colorbar_title=cfg.rtdose_colorbar_title,
                     filename=os.path.join(figures_dir, patient_id, 'rtdose.png'),
                     vmin=cfg.rtdose_vmin, vmax=cfg.rtdose_vmax,
                     ticks_steps=cfg.rtdose_ticks_steps)

            # segmentation_map
            # Make sure that the list of unique values in output_segmentation_map is less than or identical to the \
            # expected list of unique values, i.e. [0, 1, ..., len(cfg.structures_uncompound_list)]. +1 is for background
            # Normally those lists should be identical, but patient_id = 3573430 has no crico, esophagus_cerv and
            # thyroid, neither in CITOR nor in DLC.
            if not np.all(
                np.unique(segmentation_map_arr) == [x for x in range(1 + len(cfg.structures_uncompound_list))]):
                logger.my_print('np.unique(segmentation_map_arr): {}'.format(np.unique(segmentation_map_arr)),
                                level='warning')
                logger.my_print('[x for x in range(1 + len(cfg.structures_uncompound_list))]: {}'.format(
                    [x for x in range(1 + len(cfg.structures_uncompound_list))]), level='warning')

            plot_arr(segmentation_map_arr > 0, nr_images=cfg.plot_nr_images, figsize=figsize,
                     cmap=cfg.segmentation_cmap, colorbar_title=cfg.segmentation_colorbar_title,
                     filename=os.path.join(figures_dir, patient_id, 'segmentation.png'),
                     ticks_steps=cfg.segmentation_ticks_steps,
                     segmentation_map=segmentation_map_arr)

            # Plot all arrays in one plot
            arr_list = [ct_arr, ct_arr, rtdose_arr, segmentation_map_arr]
            # arr_list = [ct_arr, rtdose_arr, segmentation_map_arr > 0]
            cmap_list = [cfg.ct_cmap, cfg.ct_cmap, cfg.rtdose_cmap, cfg.segmentation_cmap]
            vmin_list = [None, None, 0, None]
            vmax_list = [None, None, cfg.rtdose_vmax, None]
            # Overlay settings
            overlay_list = [rtdose_arr, None, None, None]
            alpha_list = [0.3, None, None, None]
            cmap_overlay_list = [cfg.rtdose_cmap, None, None, None]
            vmin_overlay_list = [0, None, None, None]
            vmax_overlay_list = [cfg.rtdose_vmax, None, None, None]

            ticks_steps_list = [cfg.ct_ticks_steps, cfg.ct_ticks_steps, cfg.rtdose_ticks_steps,
                                cfg.segmentation_ticks_steps]
            segmentation_map_list = [segmentation_map_arr, None, None, None]
            colorbar_title_list = [None, cfg.ct_colorbar_title, cfg.rtdose_colorbar_title,
                                   cfg.segmentation_colorbar_title]

            plot_multiple_arrs(arr_list=arr_list, overlay_list=overlay_list, alpha_overlay_list=alpha_list,
                               nr_images=cfg.plot_nr_images_multiple, figsize=cfg.figsize_multiple,
                               cmap_list=cmap_list, cmap_overlay_list=cmap_overlay_list,
                               colorbar_title_list=colorbar_title_list,
                               filename=os.path.join(figures_dir, patient_id, 'all.png'),
                               vmin_list=vmin_list, vmin_overlay_list=vmin_overlay_list,
                               vmax_list=vmax_list, vmax_overlay_list=vmax_overlay_list,
                               ticks_steps_list=ticks_steps_list,
                               segmentation_map_list=segmentation_map_list)

            # Create multiple arrays plot in a single folder
            plot_multiple_arrs(arr_list=arr_list, overlay_list=overlay_list, alpha_overlay_list=alpha_list,
                               nr_images=cfg.plot_nr_images_multiple, figsize=cfg.figsize_multiple,
                               cmap_list=cmap_list, cmap_overlay_list=cmap_overlay_list,
                               colorbar_title_list=colorbar_title_list,
                               filename=os.path.join(figures_all_dir, '{}.png'.format(patient_id)),
                               vmin_list=vmin_list, vmin_overlay_list=vmin_overlay_list,
                               vmax_list=vmax_list, vmax_overlay_list=vmax_overlay_list,
                               ticks_steps_list=ticks_steps_list,
                               segmentation_map_list=segmentation_map_list)

    end = time.time()
    logger.my_print('Elapsed time: {time} seconds'.format(time=round(end - start, 3)))
    logger.my_print('DONE!')


def main():
    # main_segmentation_outside_ct()
    main_plot()


if __name__ == '__main__':
    main()



