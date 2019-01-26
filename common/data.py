
def transform_columns(df, cat_vars, cont_vars):
    for v in cat_vars:
        df[v] = df[v].astype('category').cat.as_ordered()
    for v in cont_vars:
        df[v] = df[v].fillna(0).astype('float32')


def get_embedding_sizes(cat_vars, df):
    cat_sz = [(c, len(df[c].cat.categories) + 1) for c in cat_vars]
    embedding_sizes = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
    return embedding_sizes


def get_validation_index(df, frac=0.25, random=True):
    if random:
        return df.sample(frac=frac).index
    else:
        total = len(df)
        return list(range(int(total - total*frac), total))


def lr_find(learner, start_lr=1e-4, end_lr=1):
    learner.lr_find(start_lr=start_lr, end_lr=end_lr)
    learner.sched.plot(100)
