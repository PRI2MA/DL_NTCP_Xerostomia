"""
Perform Logistic Regression on the same training-internal_validation-test split as Deep Learning model.

Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""
import json
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import data_preproc.data_preproc_config as data_preproc_config


def preprocess_baseline(df):
    """
    Preprocess baseline feature: map value=1 to 0, and map value>1 to 1.

    Args:
        df:

    Returns:

    """
    # Make sure that df type is Pandas Series
    assert type(df) == pd.core.series.Series

    # Make sure that the minimal value is 1 (i.e. there is no value 0, -1, -2, ...)
    assert df.min() == 1

    return df.apply(lambda x: 0 if x == 1 else 1)


def preprocess_label(df):
    """
    Preprocess label:
        - if label already only contains 0s and 1s, then keep those values
        - else map value=1, 2 to 0, and map value>2 to 1.

    Args:
        df: Pandas Series of the endpoint/label column

    Returns:

    """
    # Make sure that df type is Pandas Series
    assert type(df) == pd.core.series.Series

    unique_values = df.unique()

    if len(unique_values) == 2:
        # unique_values is expected to be [0, 1]
        unique_values.sort()
        assert np.all(unique_values == pd.Series([0, 1]))

        return df.apply(lambda x: x)

    elif len(unique_values) == 4:
        # unique_values is expected to be [1, 2, 3, 4]
        # Make sure that the minimal value is 1 or 2, and maximum value is 3 or 4
        # (i.e. there is no value 0, -1, -2, ... and 5, 6, 7, ...)
        assert (df.min() >= 1) and (df.min() <= 2) and (df.max() >= 3) and (df.max() <= 4)

        return df.apply(lambda x: 0 if x <= 2 else 1)

    else:
        # Invalid labels
        assert False


def run_logistic_regression(features_csv, patient_id_col, baseline_col, submodels_features, features, lr_coefficients,
                            endpoint, patient_id_length, patient_ids_json, seed, nr_of_decimals, logger):
    """
    Run logistic regression model on training (training + internal_validation) and test data.

    Note: if submodels_features and lr_coefficients are both None, then the logistic regression model will be
        directly fitted to features.

    Assumption: there are no overlapping of patient_ids between train, internal_validation and test set
        (see load_data.py).

    Args:
        features_csv:
        patient_id_col:
        baseline_col:
        submodels_features (list of lists): features of submodels. len(submodels_features) = nr_submodels.
        features (list): features of final model
        lr_coefficients (list): reusing coefficients
        endpoint (str):
        patient_id_length:
        patient_ids_json:
        seed:
        nr_of_decimals:
        logger:

    Returns:

    """
    # Check that we fit on the features directly (i.e. if submodels_features and lr_coefficients are both None),
    # fit submodels and construct final model from submodels (i.e. if submodels_features is not None) or
    # reuse coefficients (i.e. if lr_coefficients is not None)
    if (submodels_features is None) and (lr_coefficients is None):
        logger.my_print('Fitting LR model directly using features: {}.'.format(features))
    elif (submodels_features is None) and (lr_coefficients is not None):
        logger.my_print('Reusing coefficients of features: {} = {}.'.format(['intercept'] + features, lr_coefficients))
    elif (submodels_features is not None) and (lr_coefficients is None):
        logger.my_print('Refitting submodels using features: {}.'.format(submodels_features))
        logger.my_print('Constructing final model from submodels using features: {}.'.format(features))
    # elif (submodels_features is not None) and (lr_coefficients is not None):
    else:
        raise ValueError('Only one or both of submodels_features = {} (for fitting) and lr_coefficients = {} '
                         '(for reusing coefficients) should be None. If both are None, then we will directly fit '
                         'the logistic regression model on features = {}.'
                         .format(submodels_features, lr_coefficients, features))

    # Initialize variables
    model = LogisticRegression(random_state=seed)
    test_patient_ids, test_y_pred, test_y = [None] * 3

    # Load features + labels
    df = pd.read_csv(features_csv, sep=';', decimal=',')
    df.loc[:, patient_id_col] = df.loc[:, patient_id_col].apply(lambda x: '%0.{}d'.format(patient_id_length) % x)

    # Update endpoint
    df.loc[:, endpoint] = preprocess_label(df.loc[:, endpoint])

    # Load train_val_test_patient_ids.json for training-internal_validation-testing split
    patient_ids_dict_file = open(patient_ids_json)
    patient_ids_dict = json.load(patient_ids_dict_file)

    # Construct training-internal_validation-testing set
    df_train = df[df.loc[:, patient_id_col].isin(patient_ids_dict['train'])]
    df_val = df[df.loc[:, patient_id_col].isin(patient_ids_dict['val'])]
    df_test = df[df.loc[:, patient_id_col].isin(patient_ids_dict['test'])]

    assert len(df_train) == len(patient_ids_dict['train'])
    assert len(df_val) == len(patient_ids_dict['val'])
    assert len(df_test) == len(patient_ids_dict['test'])

    # Construct endpoints
    train_y = df_train.loc[:, endpoint]
    n_train = len(train_y)
    n_train_0 = sum(train_y == 0)
    n_train_1 = sum(train_y == 1)
    assert n_train == n_train_0 + n_train_1

    if submodels_features is not None:
        # Initialize variables
        nr_submodels = len(submodels_features)
        logger.my_print('Number of submodels: {}.'.format(nr_submodels))

        # Initialize list of submodel coefficients
        coeffs_list = []
        for i, features_i in enumerate(submodels_features):
            # Training set submodel i
            train_X = df_train.loc[:, features_i]

            # Fit submodel i
            model = model.fit(train_X, train_y)

            # Create dict of coefficients of submodel i
            keys = ['intercept'] + features_i
            values = np.append(model.intercept_, model.coef_)
            coeffs_i_dict = {k: v for (k, v) in zip(keys, values)}
            coeffs_list.append(coeffs_i_dict)

        logger.my_print('coeffs_list: {}'.format(coeffs_list))

        # Construct coefficients of final model
        lr_coefficients = []
        for f in ['intercept'] + features:
            coeff_i = 0
            # For-loop over coefficients of submodels
            for d_i in coeffs_list:
                if f in d_i.keys():
                    coeff_i += d_i[f]

            lr_coefficients.append(coeff_i / nr_submodels)

    logger.my_print('lr_coefficients: {}.'.format(lr_coefficients))

    # Training set final model
    train_patient_ids = df_train.loc[:, patient_id_col].tolist()
    train_X = df_train.loc[:, features]

    # Fit on training set or use pretrained coefficients
    model = model.fit(train_X, train_y)

    if lr_coefficients is not None:
        model.intercept_ = np.array([lr_coefficients[0]])
        model.coef_ = np.array([lr_coefficients[1:]])

    train_y_pred = model.predict_proba(train_X)
    train_y_pred = torch.tensor(train_y_pred)
    train_y = torch.tensor(train_y.tolist(), dtype=torch.int)

    # Internal validation set
    val_patient_ids = df_val.loc[:, patient_id_col].tolist()
    val_X = df_val.loc[:, features]
    val_y = df_val.loc[:, endpoint]
    n_val = len(val_y)
    n_val_0 = sum(val_y == 0)
    n_val_1 = sum(val_y == 1)
    assert n_val == n_val_0 + n_val_1

    val_y_pred = model.predict_proba(val_X)
    val_y_pred = torch.tensor(val_y_pred)
    val_y = torch.tensor(val_y.tolist(), dtype=torch.int)

    # Print label distribution
    logger.my_print('Training size (label=0): {}/{} ({}).'.format(n_train_0, n_train,
                                                                  round(n_train_0 / n_train, nr_of_decimals)))
    logger.my_print('Training size (label=1): {}/{} ({}).'.format(n_train_1, n_train,
                                                                  round(n_train_1 / n_train, nr_of_decimals)))
    logger.my_print('Internal validation size (label=0): {}/{} ({}).'.format(n_val_0, n_val,
                                                                             round(n_val_0 / n_val,
                                                                                   nr_of_decimals)))
    logger.my_print('Internal validation size (label=1): {}/{} ({}).'.format(n_val_1, n_val,
                                                                             round(n_val_1 / n_val,
                                                                                   nr_of_decimals)))

    # Test set
    if len(df_test) > 0:
        test_patient_ids = df_test.loc[:, patient_id_col].tolist()
        test_X = df_test.loc[:, features]
        test_y = df_test.loc[:, endpoint]
        n_test = len(test_y)
        n_test_0 = sum(test_y == 0)
        n_test_1 = sum(test_y == 1)
        assert n_test == n_test_0 + n_test_1

        test_y_pred = model.predict_proba(test_X)
        test_y_pred = torch.tensor(test_y_pred)
        test_y = torch.tensor(test_y.tolist(), dtype=torch.int)
        logger.my_print('Test size (label=0): {}/{} ({}).'.format(n_test_0, n_test,
                                                                  round(n_test_0 / n_test, nr_of_decimals)))
        logger.my_print('Test size (label=1): {}/{} ({}).'.format(n_test_1, n_test,
                                                                  round(n_test_1 / n_test, nr_of_decimals)))

    return (train_patient_ids, train_y_pred, train_y,
            val_patient_ids, val_y_pred, val_y,
            test_patient_ids, test_y_pred, test_y,
            lr_coefficients)


