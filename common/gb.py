from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from fastai.structured import *
from fastai.metrics import *

from tensorflow.python import keras


def kfold_gb(model, train_df, test_df, num_folds, path, label_col, target_col,
             feats_excluded=None, out_cols=None, stratified=False, cat_cols=[], name=None):
    print("Start tensorflow gradient boosting. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

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
                                                     monitor='val_binary_accuracy', save_best_only=True)]
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
    test_df[out_cols].to_csv(f'{path}/cnn_pred.csv', index=False)

