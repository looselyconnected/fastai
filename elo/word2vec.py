
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt

from elo.embedding import load_all_category
from gensim.models import Word2Vec, KeyedVectors
from sklearn.manifold import TSNE


# Train the user and merchant embeddings using per user merchant purchase percentage as target.
def train_word2vec_embeddings(path, debug=False):
    num_rows = 10000 if debug else None

    hist = pd.read_csv(f'{path}/historical_transactions.csv', nrows=num_rows,
                       usecols=['card_id', 'merchant_id', 'purchase_date'])
    new = pd.read_csv(f'{path}/new_merchant_transactions.csv', nrows=num_rows,
                      usecols=['card_id', 'merchant_id', 'purchase_date'])

    hist = hist[hist.merchant_id.notna()]
    new = new[new.merchant_id.notna()]

    df = pd.concat([hist, new]).sort_values(['card_id', 'purchase_date'])
    c_m_list = df.groupby('card_id')['merchant_id'].apply(list)

    model = Word2Vec(c_m_list, size=100, window=5, min_count=1, workers=4)
    model.save(f"{path}/models/word2vec.model")
    model.wv.save(f"{path}/models/word2vec.wv")

    return


# Load the word2vec embeddings and add them to the merchant_df by matching against the correct merchant_id
# Returns the embedding column names
def load_word2vec_embeddings(path):
    wv = KeyedVectors.load(f"{path}/models/word2vec.wv", mmap='r')
    # wv_dict = {}
    # for k in wv.vocab.keys:
    #     wv_dict[k] = list(wv[k])
    emb_cols = [f'v{i}' for i in range(wv.vector_size)]
    wv_list = [(k, wv.vocab[k].index) for k in wv.vocab.keys()]
    merchant_df = pd.DataFrame(wv_list, columns=['merchant_id', 'id_index'])
    #merchant_df.sort_values(by=['id_index'], inplace=True)
    merchant_df.set_index(['id_index'], inplace=True, drop=True)
    vector_df = pd.DataFrame(wv.vectors, columns=emb_cols)
    vector_df.index.name = 'id_index'
    merchant_df = merchant_df.merge(vector_df, on='id_index')

    # for i in range(wv.vector_size):
    #     merchant_df[f'v{i}'] = merchant_df.merchant_id.apply(lambda x: wv[x][i])
    #     if i > 5:
    #         break
    return merchant_df.set_index(['merchant_id'], drop=True)


def plot_word2vec_embeddings(path):
    num_rows = 10000
    merchant_df = pd.read_csv(f'{path}/merchants.csv', nrows=num_rows)
    emb_df = load_word2vec_embeddings(path)
    merchant_df = merchant_df.merge(emb_df, on=['merchant_id'], how='left')

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(merchant_df[emb_cols].values)
    merchant_df['x-tsne'] = tsne_results[:, 0]
    merchant_df['y-tsne'] = tsne_results[:, 1]
    plt.scatter(x=merchant_df['x-tsne'], y=merchant_df['y-tsne'], c=merchant_df.state_id)

    return
