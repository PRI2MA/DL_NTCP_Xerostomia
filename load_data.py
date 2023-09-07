"""
Note: exclude patients in exclude_patients.csv
"""
import os
import json
import math
import torch
import random
import numpy as np
import pandas as pd
from monai.data import Dataset, CacheDataset, PersistentDataset, DataLoader, ThreadDataLoader
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CenterSpatialCropd,
    DeleteItemsd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    Resized,
    RandSpatialCropd,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGridDistortiond,
    RandRotated,
    RandZoomd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    MapLabelValued,
    ToDeviced,
    CopyItemsd,
)

import config
import misc
import data_preproc.data_preproc_config as data_preproc_config
from misc import get_df_stats, stratified_sampling_split
from data_preproc.data_preproc_functions import set_default, create_folder_if_not_exists


def perform_stratified_sampling(df, frac, strata_groups, split_col, output_path_filename, seed, logger):
    """
    Create stratified samples for validation and test set.

    Important, apply this function inside get_files() (because get_files() will be run earlier if we perform CV)

    Source: https://github.com/pandas-dev/pandas/blob/v1.4.2/pandas/core/groupby/groupby.py#L3687-L3813
    Source: sample.sample(): https://github.com/pandas-dev/pandas/blob/v1.4.2/pandas/core/sample.py
    Source: random_state.choice(): https://github.com/numpy/numpy/blob/main/numpy/random/_generator.pyx#L601-L862

    Args:
        df:
        frac:
        strata_groups:
        split_col:
        output_path_filename:
        seed:
        logger:

    Returns: dataframe with column 'Split' indicating whether the patient_id belongs to 'train', 'val' or 'test'

    """
    # Train - Val split
    df_train_val = df[df[split_col] == 'train_val']
    df_test = df[df[split_col] == 'test']
    df_train, df_val = stratified_sampling_split(df=df_train_val, groups=strata_groups, frac=frac, seed=seed)

    assert len(df) == len(df_train.index.tolist() + df_val.index.tolist() + df_test.index.tolist())
    assert len(df) == len(set(df_train.index.tolist() + df_val.index.tolist() + df_test.index.tolist()))

    # Create column indicating whether the patient_id belongs to 'train', 'val' or 'test'
    df_train[split_col] = 'train'
    df_val[split_col] = 'val'
    df_test[split_col] = 'test'

    # Save df_split
    df_split = pd.concat([df_train, df_val, df_test]).sort_index()
    df_split.to_csv(output_path_filename, sep=';', index=False)


def get_files(sampling_type, features, filename_stratified_sampling_test_csv, filename_stratified_sampling_full_csv,
              perform_stratified_sampling_full, seed, logger):
    """
    Fetch data files and return the training, internal validation and test files. The data directory should be
    organized as follows:
    - data_dir:
        - 0
            - foo.npy
            - foofoo1.npy
                ...
            - foofoofoofooN.npy
        - 1
            - bar.npy
            - barbar1.npy
                ...
            - barbarbarbarN.npy

    Args:
        sampling_type:
        features:
        filename_stratified_sampling_test_csv:
        filename_stratified_sampling_full_csv:
        perform_stratified_sampling_full:
        seed:
        logger:

    Returns:
        Training, internal validation and test files.
    """
    # Initialize variables
    filename_exclude_patients_csv = data_preproc_config.filename_exclude_patients_csv
    # filename_features_csv = data_preproc_config.filename_features_csv
    patient_id_length = data_preproc_config.patient_id_length
    patient_id_col = data_preproc_config.patient_id_col
    data_dir = config.data_dir
    perform_test_run = config.perform_test_run
    train_frac = config.train_frac
    val_frac = config.val_frac
    strata_groups = config.strata_groups
    split_col = config.split_col

    # Load list of patients to be excluded
    # try:
    df_exclude_patients = pd.read_csv(filename_exclude_patients_csv, sep=';', header=None, index_col=0)
    # Remove nan's
    df_exclude_patients = df_exclude_patients[~pd.isnull(df_exclude_patients.index.values)]
    exclude_patients = df_exclude_patients.index.values.astype('int64')
    # except:
    #     exclude_patients = []
    #     logger.my_print('Cannot open or read {}. Setting exclude_patients = {}.'.
    #                     format(filename_exclude_patients_csv, exclude_patients), level='warning')

    logger.my_print('Number of patients excluded: {}.'.format(len(exclude_patients)))
    for p in exclude_patients:
        logger.my_print('Patient_id = {} will be excluded.'.format('%0.{}d'.format(patient_id_length) % p))

    # Load features
    df_features = pd.read_csv(os.path.join(data_dir, filename_stratified_sampling_test_csv), sep=';', decimal=',')
    df_features[patient_id_col] = ['%0.{}d'.format(patient_id_length) % int(x) for x in df_features[patient_id_col]]

    # Create list of images and labels
    images_list, labels_list, features_list, patient_ids_list = list(), list(), list(), list()
    labels_unique = [x for x in os.listdir(data_dir) if not x.endswith('.csv')]
    # sort() to make sure that different platforms use the same order --> required for random.shuffle() later
    labels_unique.sort()

    # Get PatientIDs
    patient_ids_list = df_features[patient_id_col].values.tolist()
    # sort() to make sure that different platforms use the same order --> required for random.shuffle() later
    patient_ids_list.sort()
    for patient_id in patient_ids_list:
        # Patient's features data
        df_features_i = df_features[df_features[patient_id_col] == patient_id]
        # Features
        features_list += [[float(str(x).replace(',', '.')) for x in y] for y in
                          df_features_i[features].values.tolist()]
        # Endpoint
        labels_list.append(int(df_features_i[data_preproc_config.endpoint]))

    assert len(patient_ids_list) == len(features_list) == len(labels_list)
    # Note: '0' in front of string is okay: int('0123') will become 123
    data_dicts = [
        {'ct': os.path.join(data_dir, str(label_name), patient_id, data_preproc_config.filename_ct_npy),
         'rtdose': os.path.join(data_dir, str(label_name), patient_id, data_preproc_config.filename_rtdose_npy),
         'segmentation_map': os.path.join(data_dir, str(label_name), patient_id,
                                          data_preproc_config.filename_segmentation_map_npy),
         'features': feature_name,
         'label': label_name,
         'patient_id': patient_id}
        for patient_id, feature_name, label_name in zip(patient_ids_list, features_list, labels_list)
        if int(patient_id) not in exclude_patients
    ]

    # Whether to perform random split or to perform stratified sampling
    if sampling_type == 'random':
        # Select random subset for testing
        if perform_test_run:
            data_dicts = data_dicts[:config.n_samples]

        # Training-internal_validation-test data split
        patients_test = df_features[df_features[split_col] == 'test'][patient_id_col].tolist()
        train_val_dict = [x for x in data_dicts if x['patient_id'] not in patients_test]
        n_train = round(len(train_val_dict) * train_frac)

        train_dict = train_val_dict[:n_train]
        val_dict = train_val_dict[n_train:]
        test_dict = [x for x in data_dicts if x['patient_id'] in patients_test]

    elif sampling_type == 'stratified':
        # Stratified sampling
        path_filename = os.path.join(data_dir, filename_stratified_sampling_full_csv)
        if not os.path.isfile(path_filename) or perform_stratified_sampling_full:
            # Create stratified_sampling.csv if file does not exists, or recreate if requested
            logger.my_print('Creating {}.'.format(path_filename))
            # Create stratified samples for train and validation set
            perform_stratified_sampling(df=df_features, frac=train_frac / (train_frac + val_frac),
                                        strata_groups=strata_groups, split_col=split_col,
                                        output_path_filename=path_filename, seed=seed, logger=logger)

        # Load list of patients with split info
        df_split = pd.read_csv(path_filename, sep=';')
        df_split[patient_id_col] = ['%0.{}d'.format(patient_id_length) % int(x) for x in df_split[patient_id_col]]

        # Exclude patients
        df_split = df_split[~df_split[patient_id_col].astype(np.int64).isin(exclude_patients)]

        # Make sure that files in the dataset folders comprehend with the label in filename_stratified_sampling_test_csv
        for l in labels_unique:
            # patient_ids_l = [x.replace('.npy', '') for x in os.listdir(os.path.join(data_dir, l))]
            patient_ids_l = os.listdir(os.path.join(data_dir, l))
            patient_ids_l = [x for x in patient_ids_l if
                             int(x) not in exclude_patients and x in df_split[patient_id_col].tolist()]
            for p in patient_ids_l:
                df_i = df_split[df_split[patient_id_col] == p]
                assert int(df_i[data_preproc_config.endpoint].values[0]) == int(l)

        # Split
        total_size = len(df_split)
        df_train = df_split[df_split[split_col] == 'train']
        df_val = df_split[df_split[split_col] == 'val']
        df_test = df_split[df_split[split_col] == 'test']

        # Print stats
        get_df_stats(df=df_split, groups=strata_groups, mode='full', frac=1, total_size=total_size, logger=logger)
        get_df_stats(df=df_train, groups=strata_groups, mode='train', frac=train_frac, total_size=total_size,
                     logger=logger)
        get_df_stats(df=df_val, groups=strata_groups, mode='val', frac=val_frac, total_size=total_size, logger=logger)
        get_df_stats(df=df_test, groups=strata_groups, mode='test', frac=1 - train_frac - val_frac,
                     total_size=total_size, logger=logger)

        train_dict, val_dict, test_dict = list(), list(), list()
        for d_dict in data_dicts:
            patient_id = d_dict['patient_id']
            df_split_i = df_split[df_split[patient_id_col] == patient_id]
            assert len(df_split_i) == 1
            split_i = df_split_i['Split'].values[0]
            if split_i == 'train':
                train_dict.append(d_dict)
            elif split_i == 'val':
                val_dict.append(d_dict)
            elif split_i == 'test':
                test_dict.append(d_dict)
            else:
                raise ValueError('Invalid split_i: {}.'.format(split_i))

        assert len(df_features) == len(data_dicts) + len(exclude_patients) == len(df_split) + len(exclude_patients) == \
               len(train_dict) + len(val_dict) + len(test_dict) + len(exclude_patients)

        # Select random subset for testing
        if perform_test_run:
            n_samples = config.n_samples
            n_train = round(n_samples * train_frac)
            n_val = math.ceil(n_samples * val_frac)
            n_test = n_samples - n_train - n_val

            random.shuffle(train_dict)
            random.shuffle(val_dict)
            random.shuffle(test_dict)

            train_dict = train_dict[:n_train]
            val_dict = val_dict[:n_val]
            test_dict = test_dict[:n_test]

    else:
        raise ValueError('Invalid sampling_type: {}.'.format(sampling_type))

    return train_dict, val_dict, test_dict


def get_files_stats(train_dict, val_dict, test_dict, features, logger):
    nr_of_decimals = config.nr_of_decimals

    # logger.my_print('train_dict: {}.'.format(train_dict))
    # logger.my_print('val_dict: {}.'.format(val_dict))
    # logger.my_print('test_dict: {}.'.format(test_dict))
    logger.my_print('Extra features for Deep Learning model: {}.'.format(features))

    n_train = len(train_dict)
    n_val = len(val_dict)
    n_test = len(test_dict)
    n_features = len(features)

    logger.my_print('Total number of data samples: {}.'.format(n_train + n_val + n_test))

    # Print label distribution
    n_train_0 = sum([True for x in train_dict if x['label'] == 0])
    n_train_1 = sum([True for x in train_dict if x['label'] == 1])
    n_val_0 = sum([True for x in val_dict if x['label'] == 0])
    n_val_1 = sum([True for x in val_dict if x['label'] == 1])
    n_test_0 = sum([True for x in test_dict if x['label'] == 0])
    n_test_1 = sum([True for x in test_dict if x['label'] == 1])

    if n_train > 0:
        logger.my_print('Training size (label=0): {}/{} ({}).'.format(n_train_0, n_train,
                                                                      round(n_train_0 / n_train, nr_of_decimals)))
        logger.my_print('Training size (label=1): {}/{} ({}).'.format(n_train_1, n_train,
                                                                      round(n_train_1 / n_train, nr_of_decimals)))
    if n_val > 0:
        logger.my_print('Internal validation size (label=0): {}/{} ({}).'.format(n_val_0, n_val,
                                                                                 round(n_val_0 / n_val,
                                                                                       nr_of_decimals)))
        logger.my_print('Internal validation size (label=1): {}/{} ({}).'.format(n_val_1, n_val,
                                                                                 round(n_val_1 / n_val,
                                                                                       nr_of_decimals)))
    if n_test > 0:
        logger.my_print('Test size (label=0): {}/{} ({}).'.format(n_test_0, n_test,
                                                                  round(n_test_0 / n_test, nr_of_decimals)))
        logger.my_print('Test size (label=1): {}/{} ({}).'.format(n_test_1, n_test,
                                                                  round(n_test_1 / n_test, nr_of_decimals)))

    return n_train_0, n_train_1, n_val_0, n_val_1, n_test_0, n_test_1, n_features


def save_patient_ids(train_dict, val_dict, test_dict, filename):
    """
    Save patient_ids in train_dict, val_dict and test_dict to json file. This information can be used
    for logistic regression.

    Args:
        train_dict:
        val_dict:
        test_dict:
        filename:

    Returns:

    """
    # Initialize variables
    patient_ids_dict = dict()
    files = [train_dict, val_dict, test_dict]
    tags = ['train', 'val', 'test']

    # Create dictionary of patient_ids
    for file, tag in zip(files, tags):
        patient_id = [x['patient_id'] for x in file]
        patient_id.sort()
        patient_ids_dict[tag] = patient_id

    # Make sure that there are no overlapping of patient_ids between train, internal_validation and test set
    assert len(misc.intersection(patient_ids_dict['train'], patient_ids_dict['val'])) == 0
    assert len(misc.intersection(patient_ids_dict['train'], patient_ids_dict['test'])) == 0
    assert len(misc.intersection(patient_ids_dict['val'], patient_ids_dict['test'])) == 0

    # Save patient_ids_dict
    with open(os.path.join(config.exp_dir, filename), 'w') as f:
        json.dump(patient_ids_dict, f, default=set_default)


def get_normalization_dataloader(train_dict, val_transforms):
    """
    Construct PyTorch Dataset for computing normalization parameters (mean, std), using training data with
    val_transforms.

    Args:
        train_dict:
        val_transforms:

    Returns:

    """
    dataset_type = config.dataset_type
    cache_rate = config.cache_rate
    num_workers = config.num_workers
    cache_dir = config.cache_dir
    dataloader_type = config.dataloader_type
    train_size = len(train_dict)
    update_dict = None

    # Define Dataset class
    if dataset_type in ['standard', None]:
        ds_class = Dataset
    elif dataset_type == 'cache':
        ds_class = CacheDataset
        update_dict = {'cache_rate': cache_rate, 'num_workers': num_workers}
    elif dataset_type == 'persistent':
        ds_class = PersistentDataset
        update_dict = {'cache_dir': cache_dir}
        create_folder_if_not_exists(cache_dir)
    else:
        raise ValueError('Invalid dataset_type: {}.'.format(dataset_type))

    # Define Dataset function arguments
    ds_args_dict = {'data': train_dict, 'transform': val_transforms}

    # Update Dataset function arguments based on type of Dataset class
    if update_dict is not None:
        ds_args_dict.update(update_dict)

    # Initialize Dataset
    ds = ds_class(**ds_args_dict)

    # Define DataLoader class
    if dataloader_type in ['standard', None]:
        dl_class = DataLoader
    elif dataloader_type == 'thread':
        dl_class = ThreadDataLoader
    else:
        raise ValueError('Invalid dataloader_type: {}.'.format(dataloader_type))

    dl_args_dict = {'dataset': ds, 'batch_size': 1, 'shuffle': False, 'num_workers': int(num_workers // 2),
                    'drop_last': False}

    # Initialize DataLoader
    dl = dl_class(**dl_args_dict) if train_size > 0 else None

    return dl


def get_mean_and_std(dataloader):
    """
    Calculate mean and std of the two (CT and RTDOSE) channels in each batch and average them at the end.

    Source: https://xydida.com/2022/9/11/ComputerVision/Normalize-images-with-transform-in-pytorch-dataloader/
    Source: https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c

    Args:
        dataloader:

    Returns:
        mean: list with length = 2 (one mean value for each channel)
        std: list with length = 2 (one std value for each channel)

    """
    # Approach 1 (
    mean = torch.zeros(3)
    # median = torch.zeros(3)
    std = torch.zeros(3)
    for batch_data in dataloader:
        inputs = batch_data['ct_dose_seg']
        for d in range(3):
            mean[d] += inputs[:, d, :, :].mean()
            # median[d] += inputs[:, d, :, :].median()
            std[d] += inputs[:, d, :, :].std()
    mean.div_(len(dataloader))
    # median.div_(len(dataloader))
    std.div_(len(dataloader))

    # Approach 2 (may not work for median)
    # var[X] = E[X**2] - E[X]**2
    # channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
    # for batch_data in dataloader:
    #     inputs = batch_data['ct_dose_seg']
    #     # Use 'weight' in case
    #     this_batch_size = inputs.size()[0]
    #     weight = this_batch_size / dataloader.batch_size
    #     channels_sum += weight * torch.mean(inputs, dim=[0, 2, 3, 4])
    #     channels_sqrd_sum += weight * torch.mean(inputs ** 2, dim=[0, 2, 3, 4])
    #     num_batches += weight
    #
    # mean = channels_sum / num_batches
    # std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return list(mean.numpy()), list(std.numpy())


def get_transforms(perform_data_aug, modes_3d, modes_2d, data_aug_p, data_aug_strength, rand_cropping_size, input_size,
                   resize_mode, seg_orig_labels, seg_target_labels, device, logger):
    """
    Transforms for training, internal validation and test data.

    Source: https://github.com/Project-MONAI/tutorials/blob/master/modules/3d_image_transforms.ipynb
    Source: https://docs.monai.io/en/latest/transforms.html?highlight=Rand3DElastic#monai.transforms.Rand3DElastic

    Note: MONAI documentation describes that the transformers require 3D data to have shape (nchannels, H, W, D), but
        shapes (1, H, W, D) and (H, W, D) results in same augmented data, i.e. shape (H, W, D) is fine.

    LoadImaged: https://docs.monai.io/en/latest/transforms.html?highlight=loadimaged#monai.transforms.LoadImage
    EnsureTyped: https://docs.monai.io/en/latest/transforms.html?highlight=ensuretyped#monai.transforms.EnsureType
    Resized: https://docs.monai.io/en/latest/transforms.html?highlight=resized#monai.transforms.Resize
    ToDeviced: Move PyTorch Tensor to the specified device. It can help cache data into GPU and execute on GPU directly.
        CacheDataset caches the transform results until ToDeviced, so it is in GPU memory. Then in every epoch, the
        program fetches cached data from GPU memory and only execute the transforms after `ToDeviced` on GPU directly.
    Source: https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_model_training_guide.md
    Source: https://docs.monai.io/en/latest/transforms.html?highlight=todeviced#monai.transforms.ToDevice

    Args:
        perform_data_aug:
        modes_3d (list): interpolation modes for 3D transformations (e.g. zoom)
        modes_2d (list): interpolation modes for 2D transformation (e.g. rotations)
        # range_value: original_max_value - original_min_value, required for fair RandGaussianNoise()
        p: probability
        data_aug_strength: strength of data augmentation: data_aug_strength=0 is no data aug (except flipping)
        rand_cropping_size:
        input_size:
        resize_mode:
        seg_orig_labels:
        seg_target_labels:
        device:
        logger:

    Returns:

    """
    to_device = config.to_device
    image_keys = config.image_keys
    concat_key = config.concat_key
    # Define variables for exceptions
    align_corners_exception_3d = [True if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None for mode in
                                  modes_3d]
    align_corners_exception_2d = [True if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None for mode in
                                  modes_2d]
    # CT
    range_value_ct = config.ct_a_max - config.ct_a_min
    # RTDOSE
    range_value_rtdose = config.rtdose_a_max - config.rtdose_a_min
    # Segmentation_map
    # values_filter = [data_preproc_config.structures_values[x] for x in config.segmentation_structures]

    logger.my_print('To_device: {}.'.format(to_device))

    # Define generic transforms
    generic_transforms = Compose([
        LoadImaged(keys=image_keys),
        EnsureTyped(keys=image_keys + ['features', 'label'], data_type='tensor'),
        # Clip
        ScaleIntensityRanged(keys=['ct'],
                             a_min=config.ct_a_min, a_max=config.ct_a_max,
                             b_min=config.ct_a_min, b_max=config.ct_a_max,
                             clip=config.ct_clip),
        ScaleIntensityRanged(keys=['rtdose'],
                             a_min=config.rtdose_a_min, a_max=config.rtdose_a_max,
                             b_min=config.rtdose_a_min, b_max=config.rtdose_a_max,
                             clip=config.rtdose_clip),
        # Useful for creating figures such as attention maps
        # CopyItemsd(keys=['segmentation_map'], names=['segmentation_map_original'], times=1),
        MapLabelValued(keys=['segmentation_map'], orig_labels=seg_orig_labels, target_labels=seg_target_labels),
        ConcatItemsd(keys=image_keys, name=concat_key, dim=0),
        DeleteItemsd(keys=image_keys),
    ])

    # Define training transforms
    train_transforms = generic_transforms

    # Define internal validation and test transforms
    val_transforms = generic_transforms

    # Data augmentation
    if perform_data_aug:
        train_transforms = Compose([
            train_transforms,
            RandSpatialCropd(keys=[concat_key], roi_size=rand_cropping_size, random_center=True, random_size=False),
            RandFlipd(keys=[concat_key], prob=data_aug_p, spatial_axis=-1),  # 3D: (num_channels, H[, W, …, ])
            RandAffined(keys=[concat_key], prob=data_aug_p,
                        translate_range=(7 * data_aug_strength, 7 * data_aug_strength, 7 * data_aug_strength),
                        padding_mode='border', mode=modes_2d),  # 3D: (num_channels, H, W[, D])
            RandAffined(keys=[concat_key], prob=data_aug_p,
                        scale_range=(0.07 * data_aug_strength, 0.07 * data_aug_strength, 0.07 * data_aug_strength),
                        padding_mode='border', mode=modes_2d),  # 3D: (num_channels, H, W[, D])
            # RandZoomd(keys=concat_key, prob=data_aug_p, min_zoom=[1] + [1 - 0.07 * data_aug_strength] * 2,
            #           max_zoom=[1] + [1 + 0.07 * data_aug_strength] * 2, align_corners=align_corners_exception_3d,
            #           padding_mode='edge', mode=modes_3d),  # Is similar to  RandAffined with scale_range param.
            # 3D: (nchannels, H, W, D). Only `trilinear` possible for 5D input data
            RandRotated(keys=[concat_key], prob=data_aug_p, range_x=(np.pi / 24) * data_aug_strength,
                        align_corners=align_corners_exception_2d, padding_mode='border', mode=modes_2d),
            # 3D: (nchannels, H, W, D). Only `bicubic`, `nearest` and `bilinear` possible. `Trilinear` not available because
            # the rotation is done in 2 dimensions, so that every slice is rotated by the same amount.
            # Rand3DElasticd(keys=[concat_key], prob=data_aug_p, sigma_range=(5, 8), magnitude_range=(0, round(1 * data_aug_strength)),
            #                padding_mode='border', mode=modes_3d),  # 3D: (nchannels, H, W, D)
            # RandShiftIntensityd(keys=[concat_key], prob=data_aug_p,
            #                     offsets=((0, 0.05 * (config.ct_a_max - config.ct_a_min) * data_aug_strength),
            #                              (0, 0.05 * (config.rtdose_a_max - config.rtdose_a_min) * data_aug_strength),
            #                              (0, 0.05 * data_aug_strength))
            #                     ),
            # RandGaussianNoised(keys=['ct'], prob=data_aug_p, mean=0.0, std=(0.1 / (range_value_ct ** (1 / 2))) * data_aug_strength),
            # RandGaussianNoised(keys=['rtdose'], prob=data_aug_p, mean=0.0, std=(0.1 / (range_value_rtdose ** (1 / 2))) * data_aug_strength),
        ])

    # Resize images
    if list(rand_cropping_size) != list(input_size) or not perform_data_aug:
        train_transforms = Compose([
            train_transforms,
            CenterSpatialCropd(keys=[concat_key], roi_size=input_size),
        ])

    val_transforms = Compose([
        val_transforms,
        CenterSpatialCropd(keys=[concat_key], roi_size=input_size),
    ])

    # To device
    if to_device:
        train_transforms = Compose([
            train_transforms,
            ToDeviced(keys=[concat_key], device=device),
        ])

        val_transforms = Compose([
            val_transforms,
            ToDeviced(keys=[concat_key], device=device),
        ])

    # Flatten transforms
    train_transforms = train_transforms.flatten()
    val_transforms = val_transforms.flatten()

    # Print transforms
    for mode, t in zip(['Train', 'Validation'], [train_transforms, val_transforms]):
        logger.my_print('{} transforms:'.format(mode))
        for i in t.transforms:
            logger.my_print('\t{}, keys={}'.format(i.__class__, i.keys))

    return train_transforms, val_transforms


def get_dataloaders(train_dict, val_dict, test_dict, train_transforms, val_transforms, test_transforms,
                    batch_size, use_sampler, drop_last, logger):
    """
    Construct PyTorch Dataset object, and then DataLoader.

    CacheDataset: caches data in RAM storage. By caching the results of deterministic / non-random preprocessing
        transforms, it accelerates the training data pipeline.
    PersistentDataset: caches data in disk storage instead of RAM storage.
    Source: https://docs.monai.io/en/stable/data.html

    Args:
        train_dict:
        val_dict:
        test_dict:
        train_transforms:
        val_transforms:
        test_transforms:
        batch_size:
        use_sampler:
        drop_last:
        logger: logger

    Returns:

    """
    dataset_type = config.dataset_type
    cache_rate = config.cache_rate
    num_workers = config.num_workers
    cache_dir = config.cache_dir
    dataloader_type = config.dataloader_type
    train_size = len(train_dict)
    val_size = len(val_dict)
    test_size = len(test_dict)
    update_dict = None

    logger.my_print('Dataset type: {}.'.format(dataset_type))
    logger.my_print('Dataloader type: {}.'.format(dataloader_type))

    # Define Dataset class
    if dataset_type in ['standard', None]:
        ds_class = Dataset
    elif dataset_type == 'cache':
        ds_class = CacheDataset
        update_dict = {'cache_rate': cache_rate, 'num_workers': num_workers}
    elif dataset_type == 'persistent':
        ds_class = PersistentDataset
        update_dict = {'cache_dir': cache_dir}
        create_folder_if_not_exists(cache_dir)
    else:
        raise ValueError('Invalid dataset_type: {}.'.format(dataset_type))

    # Define Dataset function arguments
    train_ds_args_dict = {'data': train_dict, 'transform': train_transforms}
    val_ds_args_dict = {'data': val_dict, 'transform': val_transforms}
    test_ds_args_dict = {'data': test_dict, 'transform': test_transforms}

    # Update Dataset function arguments based on type of Dataset class
    if update_dict is not None:
        train_ds_args_dict.update(update_dict)
        val_ds_args_dict.update(update_dict)
        test_ds_args_dict.update(update_dict)

    # Initialize Dataset
    train_ds = ds_class(**train_ds_args_dict)
    val_ds = ds_class(**val_ds_args_dict)
    test_ds = ds_class(**test_ds_args_dict)

    # Define DataLoader class
    if dataloader_type in ['standard', None]:
        dl_class = DataLoader
    elif dataloader_type == 'thread':
        dl_class = ThreadDataLoader
    else:
        raise ValueError('Invalid dataloader_type: {}.'.format(dataloader_type))

    # Define Dataloader function arguments
    # Shuffle is not necessary for val_dl and test_dl, but shuffle can be useful for plotting random patients in main.py
    # Weighted random sampler
    if use_sampler:
        # Source: https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader
        logger.my_print('Using WeightedRandomSampler.')
        shuffle = False

        label_raw_train = np.array([x['label'] for x in train_dict])
        # len(weights) = num_classes
        weights = 1 / np.array([np.count_nonzero(1 - label_raw_train), np.count_nonzero(label_raw_train)])
        samples_weight = np.array([weights[t] for t in label_raw_train])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        shuffle = True
        sampler = None

    logger.my_print('Train dataloader arguments.')
    logger.my_print('\tBatch_size: {}.'.format(batch_size))
    logger.my_print('\tShuffle: {}.'.format(shuffle))
    logger.my_print('\tSampler: {}.'.format(sampler))
    logger.my_print('\tNum_workers: {}.'.format(num_workers))
    logger.my_print('\tDrop_last: {}.'.format(drop_last))
    train_dl_args_dict = {'dataset': train_ds, 'batch_size': batch_size, 'shuffle': shuffle, 'sampler': sampler,
                          'num_workers': num_workers, 'drop_last': drop_last}
    val_dl_args_dict = {'dataset': val_ds, 'batch_size': 1, 'shuffle': False, 'num_workers': int(num_workers // 2),
                        'drop_last': False}
    test_dl_args_dict = {'dataset': test_ds, 'batch_size': 1, 'shuffle': False, 'num_workers': int(num_workers // 2),
                         'drop_last': False}

    # Initialize DataLoader
    train_dl = dl_class(**train_dl_args_dict) if train_size > 0 else None
    val_dl = dl_class(**val_dl_args_dict) if val_size > 0 else None
    test_dl = dl_class(**test_dl_args_dict) if test_size > 0 else None

    return train_dl, val_dl, test_dl, train_ds, dl_class, train_dl_args_dict


def main(train_dict, val_dict, test_dict, features, perform_data_aug, data_aug_p, data_aug_strength, rand_cropping_size,
         input_size, resize_mode, seg_orig_labels, seg_target_labels, batch_size, use_sampler, fold, drop_last,
         sampling_type, filename_stratified_sampling_test_csv, filename_stratified_sampling_full_csv,
         perform_stratified_sampling_full, seed, device, logger):
    """
    Loads datasets and returns the batches corresponding to the given parameters.

    Args:
        train_dict:
        val_dict:
        test_dict:
        features:
        perform_data_aug
        data_aug_p:
        data_aug_strength:
        rand_cropping_size
        input_size:
        resize_mode:
        seg_orig_labels:
        seg_target_labels:
        batch_size:
        use_sampler:
        fold:
        drop_last:
        sampling_type:
        filename_stratified_sampling_test_csv:
        filename_stratified_sampling_full_csv:
        perform_stratified_sampling_full:
        seed:
        device:
        logger:

    Returns:

    """
    # Initialize variables
    norm_mean, norm_std = None, None

    # Make sure that they are no None's, or all None's (i.e. whether we perform cross-validation or not)
    nr_nones = (train_dict is None) + (val_dict is None) + (test_dict is None)
    assert nr_nones in [0, 3]

    # Get training, internal validation and test files
    if nr_nones == 3:
        train_dict, val_dict, test_dict = get_files(
            sampling_type=sampling_type, features=features,
            filename_stratified_sampling_test_csv=filename_stratified_sampling_test_csv,
            filename_stratified_sampling_full_csv=filename_stratified_sampling_full_csv,
            perform_stratified_sampling_full=perform_stratified_sampling_full,
            seed=seed, logger=logger)
    train_0, train_1, val_0, val_1, test_0, test_1, n_features = get_files_stats(train_dict, val_dict, test_dict,
                                                                                 features, logger)

    # Save list of patient_ids in training, internal validation and test
    save_patient_ids(train_dict, val_dict, test_dict,
                     filename=config.filename_train_val_test_patient_ids_json.format(fold))

    # Define training, internal validation and test transforms
    modes_3d = [config.ct_dose_seg_interpol_mode_3d]
    modes_2d = [config.ct_dose_seg_interpol_mode_2d]
    train_transforms, val_transforms = get_transforms(perform_data_aug=perform_data_aug, modes_3d=modes_3d,
                                                      modes_2d=modes_2d, data_aug_p=data_aug_p,
                                                      data_aug_strength=data_aug_strength,
                                                      rand_cropping_size=rand_cropping_size, input_size=input_size,
                                                      resize_mode=resize_mode, seg_orig_labels=seg_orig_labels,
                                                      seg_target_labels=seg_target_labels, device=device,
                                                      logger=logger)

    # Determine normalization parameters
    train_dl_normalization = get_normalization_dataloader(train_dict=train_dict, val_transforms=val_transforms)
    norm_mean, norm_std = get_mean_and_std(dataloader=train_dl_normalization)
    logger.my_print('Normalization mean: {}'.format(norm_mean))
    logger.my_print('Normalization std: {}'.format(norm_std))
    del train_dl_normalization

    # Get training, internal validation and test dataloaders
    train_dl, val_dl, test_dl, train_ds, dl_class, train_dl_args_dict = get_dataloaders(
        train_dict=train_dict, val_dict=val_dict, test_dict=test_dict,
        train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=val_transforms,
        batch_size=batch_size, use_sampler=use_sampler, drop_last=drop_last, logger=logger)

    # Get input shape
    if train_0 + train_1 > 0:
        example_data = next(iter(train_dl))
    elif val_0 + val_1 > 0:
        example_data = next(iter(val_dl))
    elif test_0 + test_1 > 0:
        example_data = next(iter(test_dl))
    # example_input = example_data['image']
    # example_input = torch.concat([example_data['ct'], example_data['rtdose'], example_data['segmentation_map']], axis=1)
    example_input = example_data['ct_dose_seg']
    batch_size, channels, depth, height, width = example_input.shape

    return (train_dl, val_dl, test_dl, train_ds, dl_class, train_dl_args_dict, batch_size, channels, depth, height,
            width, train_0, train_1, val_0, val_1, test_0, test_1, n_features, train_dict, val_dict, test_dict,
            norm_mean, norm_std)
