import numpy as np

from titanic.data import get_tables
from titanic.data import get_embedding_sizes
from titanic.data import predict_and_save
from titanic.data import add_family_survived, add_weak_family_survived, get_family_info
from fastai.structured import *
from fastai.dataset import split_by_idx
from fastai.column_data import ColumnarModelData
np.set_printoptions(threshold=50, edgeitems=20)

PATH = 'experiment/'

def main():
    tables = get_tables(PATH, ['train', 'test'])
    train, test = tables
    val_idx = train.sample(frac=0.25).index

    family_survived = train[['LastName', 'Survived']].groupby('LastName').sum()
    add_family_survived(family_survived, test)

    # We can't train using the same family survived info in the training set. By minusing one we are using
    # the info that at least someone else in the family survived in the training set.
    # family_survived.loc[family_survived['Survived'] > 0, 'Survived'] -= 1
    family_survived['Survived'] -= 1
    add_family_survived(family_survived, train)

    # Not using last name or cabin directly right now - cardinality is too high
    cat_vars = ['Pclass', 'Sex', 'Embarked', 'Title']
    cont_vars = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySurvived']

    for table in tables:
        for v in cat_vars:
            table[v] = table[v].astype('category').cat.as_ordered()
        for v in cont_vars:
            table[v] = table[v].astype('float32')

    test['Survived'] = 0
    train = train[cat_vars + cont_vars + ['Survived']]
    test = test[cat_vars + cont_vars + ['Survived', 'PassengerId']]

    df, y, nas, mapper = proc_df(train, 'Survived', do_scale=True)
    df_test, _, nas, mapper = proc_df(test, 'Survived', do_scale=True, skip_flds=['PassengerId'],
                                      mapper=mapper, na_dict=nas)

    md = ColumnarModelData.from_data_frame('data/', val_idx, df, y.astype(np.float32), cat_flds=cat_vars,
                                           is_reg=True, is_multi=False, bs=64, test_df=df_test)
    embedding_sizes = get_embedding_sizes(cat_vars, train)

    model = md.get_learner(embedding_sizes, len(df.columns) - len(cat_vars), 0.04, 1, [10, 5], [0.001,0.01], y_range=(0, 1))
    model.summary()

    lr = 1e-3
    model.fit(lr, 10)

    # model.load('m-1')
    # model.fit(lr, 10)
    #
    predict_and_save(model, test, PATH, 'base')

    print('done')


if __name__ == "__main__":
    main()
