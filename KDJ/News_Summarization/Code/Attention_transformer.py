import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import urllib.request

data = pd.read_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess_test.csv')

##      데이터 전처리
sentences = []
for sentence in data['sentence']:
    sentences.append(sentence)
abs = []
for sentence in data['abs']:
    abs.append(sentence)

# 길이 분포 출력
text_len = [len(s.split()) for s in data['sentence']]
summary_len = [len(s.split()) for s in data['abs']]

print('텍스트의 최소 길이 : {}'.format(np.min(text_len)))
print('텍스트의 최대 길이 : {}'.format(np.max(text_len)))
print('텍스트의 평균 길이 : {}'.format(np.mean(text_len)))
print('요약의 최소 길이 : {}'.format(np.min(summary_len)))
print('요약의 최대 길이 : {}'.format(np.max(summary_len)))
print('요약의 평균 길이 : {}'.format(np.mean(summary_len)))

plt.subplot(1,2,1)
plt.boxplot(summary_len)
plt.title('abs')
plt.subplot(1,2,2)
plt.boxplot(text_len)
plt.title('sentence')
plt.tight_layout()
plt.show()

plt.title('abs')
plt.hist(summary_len, bins=40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

plt.title('sentence')
plt.hist(text_len, bins=40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
"""
sentence_max_len = 300
abs_max_len = 40

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s.split()) <= max_len):
        cnt = cnt + 1
#   print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))))

below_threshold_len(sentence_max_len, data['sentence'])
below_threshold_len(abs_max_len, data['abs'])

data = data[data['sentence'].apply(lambda x: len(x.split()) <= sentence_max_len)]
data = data[data['abs'].apply(lambda x: len(x.split()) <= abs_max_len)]
# print('전체 샘플수 :',(len(data)))

##      토큰 추가
data['decoder_input'] = data['abs'].apply(lambda x : 'sostoken '+ x)
data['decoder_target'] = data['abs'].apply(lambda x : x + ' eostoken')

encoder_input_train = np.array(data['sentence'])
decoder_input_train = np.array(data['decoder_input'])
decoder_target_train = np.array(data['decoder_target'])

##      검증셋
val_data = pd.read_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/valid_test.csv')

val_sentences = []
for sentence in val_data['sentence']:
    sentences.append(sentence)
val_abs = []
for sentence in val_data['abs']:
    abs.append(sentence)

val_data['decoder_input'] = val_data['abs'].apply(lambda x : 'sostoken '+ x)
val_data['decoder_target'] = val_data['abs'].apply(lambda x : x + ' eostoken')

encoder_input_test = np.array(val_data['sentence'])
decoder_input_test = np.array(val_data['decoder_input'])
decoder_target_test = np.array(val_data['decoder_target'])

# print('훈련 데이터의 개수 :', len(encoder_input_train))
# print('훈련 레이블의 개수 :',len(decoder_input_train))
# print('테스트 데이터의 개수 :',len(encoder_input_test))
# print('테스트 레이블의 개수 :',len(decoder_input_test))


tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/tokenizer')

def tokenizerANDpadding(list, len):
   result_list = []
   for sentence in list:
    tokenList = tokenizer.encode(sentence)
    result_list.append(tokenList)
    padList = tf.keras.preprocessing.sequence.pad_sequences(
            result_list, maxlen=len, padding='post'
        )
    return padList

encoder_input_train = tokenizerANDpadding(encoder_input_train, sentence_max_len)
encoder_input_test = tokenizerANDpadding(encoder_input_test, sentence_max_len)
decoder_input_train = tokenizerANDpadding(decoder_input_train, abs_max_len)
decoder_input_test = tokenizerANDpadding(decoder_input_test, abs_max_len)
decoder_target_train = tokenizerANDpadding(decoder_target_train, abs_max_len)
decoder_target_test = tokenizerANDpadding(decoder_target_test, abs_max_len)

# print(f'encoder_input_train : {encoder_input_train[:1]}')
# print(f'encoder_input_test : {encoder_input_test[:1]}')
# print(f'decoder_input_train : {decoder_input_train[:1]}')
# print(f'decoder_input_test : {decoder_input_test[:1]}')
# print(f'decoder_target_train : {decoder_target_train[:1]}')
# print(f'decoder_target_test : {decoder_target_test[:1]}')

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 128
hidden_size = 256

# 인코더
encoder_inputs = Input(shape=(300,))

# 인코더의 임베딩 층
enc_emb = Embedding(2**14, embedding_dim)(encoder_inputs)

# 인코더의 LSTM 1
encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True ,dropout = 0.4, recurrent_dropout = 0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# 인코더의 LSTM 2
encoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# 인코더의 LSTM 3
encoder_lstm3 = LSTM(hidden_size, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

# 디코더
decoder_inputs = Input(shape=(None,))

# 디코더의 임베딩 층
dec_emb_layer = Embedding(2**14, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)

# 디코더의 LSTM
decoder_lstm = LSTM(hidden_size, return_sequences = True, return_state = True, dropout = 0.4, recurrent_dropout=0.2)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state = [state_h, state_c])

# 디코더의 출력층
decoder_softmax_layer = Dense(2**14, activation = 'softmax')
decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs) 

# 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.summary()

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/20.%20Text%20Summarization%20with%20Attention/attention.py", filename="D:/TJ_FInal_Project/KDJ/News_Summarization/Code/attention.py")
from attention import AttentionLayer

# 어텐션 층(어텐션 함수)
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# 어텐션의 결과와 디코더의 hidden state들을 연결
decoder_concat_input = Concatenate(axis = -1, name='concat_layer')([decoder_outputs, attn_out])

# 디코더의 출력층
decoder_softmax_layer = Dense(2**14, activation='softmax')
decoder_softmax_outputs = decoder_softmax_layer(decoder_concat_input)

# 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 2)
history = model.fit(x = [encoder_input_train, decoder_input_train], y = decoder_target_train, \
          validation_data = ([encoder_input_test, decoder_input_test], decoder_target_test),
          batch_size = 256, callbacks=[es], epochs = 50)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

src_index_to_word = tokenizer.index_word # 원문 단어 집합에서 정수 -> 단어를 얻음
tar_word_to_index = tokenizer.word_index # 요약 단어 집합에서 단어 -> 정수를 얻음
tar_index_to_word = tokenizer.index_word # 요약 단어 집합에서 정수 -> 단어를 얻음
"""