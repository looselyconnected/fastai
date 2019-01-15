import numpy as np

from titanic.data import get_tables
from titanic.data import get_embedding_sizes
from titanic.data import predict_and_save
from titanic.data import add_family_survived, add_family_survived_self
from fastai.structured import *
from fastai.column_data import ColumnarModelData
np.set_printoptions(threshold=50, edgeitems=20)

PATH = 'experiment/'

def main():
    tables = get_tables(PATH, ['train', 'test'])
    train, test = tables
    val_idx = train.sample(frac=0.5).index

    family_survived = train[['LastName', 'Survived']].groupby('LastName').sum()
    add_family_survived(family_survived, test)

    # We can't train using the same family survived info in the training set because we would be cheating
    # by training on the result itself.
    family_count = train[['LastName', 'Survived']].groupby('LastName').count()
    remove_names = list(family_count[family_count['Survived'] == 1].index)
    remove_names_tuple = set([(x, ) for x in remove_names])
    train_index = train[~train[['LastName']].apply(tuple, 1).isin(remove_names_tuple)].index
    add_family_survived_self(train, train_index, val_idx)

    # Not using last name or cabin directly right now - cardinality is too high
    cat_vars = ['Pclass', 'Sex', 'Embarked', 'Title', 'FamilySurvived']
    cont_vars = ['Age', 'SibSp', 'Parch', 'Fare']

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

    md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y.astype(np.float32), cat_flds=cat_vars,
                                           is_reg=True, is_multi=False, bs=128, test_df=df_test)
    embedding_sizes = get_embedding_sizes(cat_vars, train)

    model = md.get_learner(embedding_sizes, len(df.columns) - len(cat_vars), 0.5, 1, [10, 5], [0.5, 0.5],
                           y_range=(0, 1))
    model.summary()

    lr = 1e-2
    model.fit(lr, 20)

    # model.load('m-1')
    # model.fit(lr, 10)
    #
    predict_and_save(model, test, PATH, 'base')

    print('done')


if __name__ == "__main__":
    main()
