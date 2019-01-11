from os import listdir
from os.path import realpath
import pandas as pd
import numpy as np


def get_tables(path, table_names):
    tables = [pd.read_csv(f'{path}/{fname}.csv', low_memory=False) for fname in table_names]

    for table in tables:
        name_split = table['Name'].str.split(',', expand=True)
        table['LastName'] = name_split[0]
        table['Title'] = name_split[1].str.split(' ', expand=True)[1]

        # Only keep the significant titles. Others are too few for training
        table.loc[(table['Title'] != 'Mr.') & (table['Title'] != 'Mrs.') & (table['Title'] != 'Miss.') &
                  (table['Title'] != 'Master.'), 'Title'] = None

    return tables


def get_embedding_sizes(cat_vars, df):
    cat_sz = [(c, len(df[c].cat.categories) + 1) for c in cat_vars]
    embedding_sizes = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
    return embedding_sizes


def predict_and_save(model, test, path, filename):
    pred_test = model.predict(True)
    test = test.copy()
    test.loc[:, 'Survived'] = pred_test
    test.loc[test.Survived < 0.5, 'Survived'] = int(0)
    test.loc[test.Survived >= 0.5, 'Survived'] = int(1)
    test.Survived = test.Survived.astype(int)
    test[['PassengerId', 'Survived']].to_csv(f'{path}/tmp/{filename}.csv', index=False)


def get_family_info(df):
    families = df[['LastName']].groupby('LastName').count().set_index('LastName')

    families = set(df[['LastName']].itertuples(index=False, name=None))
    family_survived = set(df[df.Survived == 1][['LastName']].itertuples(index=False, name=None))
    return families, family_survived


def add_family_survived(family_survived, df):
    df['FamilySurvived'] = np.NAN

    for index, row in df.iterrows():
        try:
            if family_survived.loc[row.LastName].Survived >= 0:
                df.loc[index, 'FamilySurvived'] = family_survived.loc[row.LastName].Survived
        except KeyError:
            pass

    # df.loc[df[['LastName']].apply(tuple, 1).isin(families), 'FamilySurvived'] = 0
    # df.loc[df[['LastName']].apply(tuple, 1).isin(family_survived), 'FamilySurvived'] = 1


def add_weak_family_survived(*a):
    for table in a:
        table.loc[((table['Age'] <= 12) | (table['Sex'] == 'female')) & (
                    table['FamilySurvived'] == True), 'WeakFamilySurvived'] = True
