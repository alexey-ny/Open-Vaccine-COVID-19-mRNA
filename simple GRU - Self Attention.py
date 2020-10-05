# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:57:02 2020

@author: alex
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import json

import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow.keras.layers as L
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tf.random.set_seed(1970)
np.random.seed(1970)

denoise = True

# This will tell us the columns we are predicting
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']

y_true = tf.random.normal((32, 68, 3))
y_pred = tf.random.normal((32, 68, 3))

data_dir = ''
train = pd.read_json(data_dir + 'train.json', lines=True)
test = pd.read_json(data_dir + 'test.json', lines=True)
test_pub = test[test["seq_length"] == 107]
test_pri = test[test["seq_length"] == 130]
sample_df = pd.read_csv(data_dir + 'sample_submission.csv')

train.columns.tolist()
example = train.iloc[0]

if denoise:
    train = train[train.signal_to_noise >= 1].reset_index(drop = True)

As = []
for id in tqdm(train["id"]):
    a = np.load(f"{data_dir}bpps/{id}.npy")
    As.append(a)
As = np.array(As)
As_max = As.max(axis=1)
As_sum = 1 - As.sum(axis=1)
As_bpps = np.transpose(np.array([As_max,As_sum]), (1,2,0))

As_pub = []
for id in tqdm(test_pub["id"]):
    a = np.load(f"{data_dir}bpps/{id}.npy")
    As_pub.append(a)
As_pub = np.array(As_pub)
As_pub_max = As_pub.max(axis=1)
As_pub_sum = 1 - As_pub.sum(axis=1)
As_pub_bpps = np.transpose(np.array([As_pub_max,As_pub_sum]), (1,2,0))

As_pri = []
for id in tqdm(test_pri["id"]):
    a = np.load(f"{data_dir}bpps/{id}.npy")
    As_pri.append(a)
As_pri = np.array(As_pri)
As_pri_max = As_pri.max(axis=1)
As_pri_sum = 1 - As_pri.sum(axis=1)
As_pri_bpps = np.transpose(np.array([As_pri_max,As_pri_sum]), (1,2,0))


def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(
        hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))

def build_model_bpps_Attn(embed_size, seq_len=107, pred_len=68, dropout=0.5, 
                sp_dropout=0.2, embed_dim=200, hidden_dim=256, n_layers=3):
    input_s = L.Input(shape=(seq_len, 3))
    input_bpps = L.Input(shape=(seq_len, 2))
    
    inputs = L.Concatenate(axis = 2)([input_s, input_bpps])
    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)
    
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3])
    )
    hidden = L.SpatialDropout1D(sp_dropout)(reshaped)
    conv1 = L.Conv1D(128, 64, 1, padding = "same", activation = None)(hidden)
    h1 = L.LayerNormalization()(conv1)
    h1 = L.LeakyReLU()(h1)
    conv2 = L.Conv1D(128, 32, 1, padding = "same", activation = None)(hidden)
    h2 = L.LayerNormalization()(conv2)
    h2 = L.LeakyReLU()(h2)
    conv3 = L.Conv1D(128, 16, 1, padding = "same", activation = None)(hidden)
    h3 = L.LayerNormalization()(conv3)
    h3 = L.LeakyReLU()(h3)
    conv4 = L.Conv1D(128, 8, 1, padding = "same", activation = None)(hidden)
    h4 = L.LayerNormalization()(conv4)
    h4 = L.LeakyReLU()(h4)
    
    hs = L.Concatenate()([h1, h2, h3, h4])

    keys = L.Dropout(0.2)(hs)    
    
    for x in range(n_layers):
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        
    hidden = L.Attention(dropout = 0.2)([hidden, keys])

    # Since we are only making predictions on the first part of each sequence, 
    # we have to truncate it
    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation='linear')(truncated)
    
    model = tf.keras.Model(inputs = [input_s, input_bpps], outputs=out)
    model.compile(tf.optimizers.Adam(), loss=MCRMSE)
    
    return model

def pandas_list_to_array(df):
    """
    Input: dataframe of shape (x, y), containing list of length l
    Return: np.array of shape (x, l, y)
    """
    
    return np.transpose(
        np.array(df.values.tolist()),
        (0, 2, 1)
    )

def preprocess_inputs(df, token2int, cols=['sequence', 'structure', 'predicted_loop_type']):
    return pandas_list_to_array(
        df[cols].applymap(lambda seq: [token2int[x] for x in seq])
    )


# We will use this dictionary to map each character to an integer
# so that it can be used as an input in keras
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

train_inputs = preprocess_inputs(train, token2int)
train_labels = pandas_list_to_array(train[pred_cols])

x_train, x_val, y_train, y_val, A_bpps_T, A_bpps_V = train_test_split(
    train_inputs, train_labels, As_bpps, test_size=.1, random_state=34, stratify=train.SN_filter)

public_df = test.query("seq_length == 107")
private_df = test.query("seq_length == 130")

public_inputs = preprocess_inputs(public_df, token2int)
private_inputs = preprocess_inputs(private_df, token2int)

model = build_model_bpps_Attn(embed_size=len(token2int), n_layers = 3)
model.summary()

history = model.fit(
    [x_train, A_bpps_T], y_train,
    validation_data=([x_val, A_bpps_V], y_val),
    batch_size=64,
    epochs=75,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
        tf.keras.callbacks.ModelCheckpoint('model.h5')
    ]
)

# Caveat: The prediction format requires the output to be the same length as the input,
# although it's not the case for the training data.
model_public = build_model_bpps_Attn(seq_len=107, pred_len=107, embed_size=len(token2int))
model_private = build_model_bpps_Attn(seq_len=130, pred_len=130, embed_size=len(token2int))

model_public.load_weights('model.h5')
model_private.load_weights('model.h5')

public_preds = model_public.predict([public_inputs, As_pub_bpps])
private_preds = model_private.predict([private_inputs, As_pri_bpps])

# For each sample, we take the predicted tensors of shape (107, 5) or (130, 5), and convert them 
# to the long format (i.e.  629×107,5  or  3005×130,5 ):

preds_ls = []

for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)
preds_df.head()

submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
submission.to_csv('submission.csv', index=False)
print('finished')

    