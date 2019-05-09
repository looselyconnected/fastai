
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from fastai.structured import *
from fastai.metrics import *

from tensorflow.python import keras

import category_encoders as ce

from common.data import get_validation_index, one_hot_encoder


def kfold_nn(model, train_df, test_df, num_folds, path, label_col, target_col,
             feats_excluded=None, out_cols=None, stratified=False, cat_cols=[], name=None, input_shape=None):
    print("Starting CNN. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    train_df[target_col] = train_df[target_col].astype(float)
    # Cross validation model
    if stratified:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=326)
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    test_df[target_col] = 0

    if feats_excluded is None:
        feats_excluded = [label_col, target_col]
    feat_cols = [f for f in train_df.columns if f not in feats_excluded]
    if out_cols is None:
        out_cols = [label_col, target_col]
    print(f'features {feat_cols}')

    train_x, train_y, nas, mapper = proc_df(train_df, target_col, do_scale=True, skip_flds=[label_col])
    test_x, _, nas, mapper = proc_df(test_df, target_col, do_scale=True, mapper=mapper, na_dict=nas,
                                     skip_flds=[label_col])

    if not input_shape:
        input_shape = (len(feat_cols), )
    train_x = np.reshape(train_x.values, (len(train_x), ) + input_shape)
    test_x = np.reshape(test_x.values, (len(test_x), ) + input_shape)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df[feat_cols], train_df[target_col])):
        print("Fold {}".format(fold))
        model_name = f'{name}-{fold}'
        model_path = f'{path}/models/{model_name}'

        try:
            model = keras.models.load_model(model_path)
            print(f'loaded model from {model_path}')
        except:
            pass

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=10),
                     keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False,
                                                     monitor='val_binary_accuracy', save_best_only=True),
                     keras.callbacks.TensorBoard(log_dir=f'{model_path}.tensorboard')]
        model.fit(train_x[train_idx], train_y[train_idx], epochs=1000, callbacks=callbacks,
                  validation_data=(train_x[valid_idx], train_y[valid_idx]))

        model = keras.models.load_model(model_path)
        current = 0
        while current < len(test_df):
            end = min(current + 10000, len(test_df))
            test_y = model.predict(test_x[current:end]).reshape(-1)
            test_df.loc[current:end-1, target_col] += (test_y / num_folds)
            current = end

    # save submission file
    test_df.reset_index(inplace=True)
    test_df[out_cols].to_csv(f'{path}/nn_pred.csv', index=False)


def train_nn(model, train_x_list, train_y_list, model_path, train_percent=0.8, random=True, monitor='accuracy'):
    try:
        model = keras.models.load_model(model_path)
        print(f'loaded model from {model_path}')
    except:
        pass

    train_len = len(train_x_list[0])
    if random:
        train_mask = np.random.choice([True, False], train_len, p=[train_percent, 1.0 - train_percent])
    else:
        train_mask = np.append([True] * int(train_percent * train_len),
                               [False] * (train_len - int(train_percent * train_len)))
    train_xs = [x[train_mask] for x in train_x_list]
    val_xs = [x[~train_mask] for x in train_x_list]
    train_ys = [x[train_mask] for x in train_y_list]
    val_ys = [x[~train_mask] for x in train_y_list]

    callbacks = [keras.callbacks.EarlyStopping(monitor=monitor, patience=10),
                 keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False,
                                                 monitor=monitor, save_best_only=True),
                 keras.callbacks.TensorBoard(log_dir=f'{model_path}.tensorboard')]
    model.fit(train_xs, train_ys, epochs=1000, callbacks=callbacks,
              validation_data=(val_xs, val_ys))

    model = keras.models.load_model(model_path)
    return model


def split_train_nn(model, train_df, test_df, path, label_col, target_col, target_as_category=False,
                   feats_excluded=None, out_cols=None, cat_cols=[], name=None,
                   train_percent=0.8, random=True, monitor='accuracy'):
    print("Starting NN. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    if feats_excluded is None:
        feats_excluded = {label_col, target_col}
    feat_cols = [f for f in train_df.columns if f not in feats_excluded]
    print(f'features {feat_cols}')

    test_df[target_col] = 0
    train_x, train_y, nas, mapper = proc_df(train_df[feat_cols + [target_col]], target_col, do_scale=True)
    test_x, _, nas, mapper = proc_df(test_df[feat_cols + [target_col]], target_col, do_scale=True, mapper=mapper, na_dict=nas)

    if target_as_category:
        # multi-label classification. We assume the target is compatible with lgbm - consequtive numbers
        target_count = int(train_df.target.max() + 1)
        one_hot_enc = ce.OneHotEncoder(cols=['target'])
        dummy_target = pd.DataFrame(np.array([i for i in range(target_count)]), columns=['target'])
        one_hot_enc.fit(dummy_target)
        train_y = one_hot_enc.transform(train_df.target).values
        target_cols = [f'{target_col}_{i}' for i in range(target_count)]

        # target_df = pd.DataFrame(train_df[target_col].astype('object'), columns=[target_col])
        # train_y_one_hot, target_cols = one_hot_encoder(target_df, prefix=target_col, nan_as_category=False)
        # train_y = train_y_one_hot.values
    else:
        train_y = train_y.astype(float)
        target_cols = [target_col]

    for target_one_hot_col in target_cols:
        test_df[target_one_hot_col] = 0

    if out_cols is None:
        out_cols = [label_col] + target_cols

    train_len = len(train_x)
    if random:
        train_mask = np.random.choice([True, False], train_len, p=[train_percent, 1.0 - train_percent])
    else:
        train_mask = np.append([True] * int(train_percent * train_len),
                               [False] * (train_len - int(train_percent * train_len)))
    train_idx = train_mask
    valid_idx = ~train_mask

    input_shape = (len(feat_cols), )
    train_x = np.reshape(train_x.values, (len(train_x), ) + input_shape)
    test_x = np.reshape(test_x.values, (len(test_x), ) + input_shape)

    model_name = f'{name}'
    model_path = f'{path}/models/{model_name}'

    try:
        model = keras.models.load_model(model_path)
        print(f'loaded model from {model_path}')
    except:
        pass

    callbacks = [keras.callbacks.EarlyStopping(monitor=monitor, patience=100),
                 keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=False,
                                                 monitor=monitor, save_best_only=True),
                 keras.callbacks.TensorBoard(log_dir=f'{model_path}.tensorboard')]
    model.fit(train_x[train_idx], train_y[train_idx], epochs=100000, callbacks=callbacks,
              validation_data=(train_x[valid_idx], train_y[valid_idx]))

    model = keras.models.load_model(model_path)
    current = 0
    test_df.reset_index(inplace=True, drop=True)
    while current < len(test_df):
        end = min(current + 10000, len(test_df))
        test_y = model.predict(test_x[current:end])
        if not target_as_category:
            test_y = test_y.reshape(-1)
        test_df.loc[current:end-1, target_cols] = test_y
        current = end

    # save submission file
    test_df[out_cols].to_csv(f'{path}/nn_pred.csv', index=False)

