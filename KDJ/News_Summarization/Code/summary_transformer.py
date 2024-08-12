import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

# 사용 가능한 GPU 수 확인
print(tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

data = pd.read_csv('D:/TJ_FInal_Project/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess.csv')
train_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

##      데이터 전처리
sentences = []
for sentence in train_data['sentence']:
    sentences.append(sentence)
abs = []
for sentence in train_data['abs']:
    abs.append(sentence)

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    sentences + abs, target_vocab_size = 2**13
)

# tokenizer.save_to_file('D:/TJ_FInal_Project/News_Summarization/Data/문서요약 텍스트/Preprocess/tokenizer')

tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('D:/TJ_FInal_Project/News_Summarization/Data/문서요약 텍스트/Preprocess/tokenizer')
# print(tokenizer.subwords[:100])

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size+1]
VOCAB_SIZE = tokenizer.vocab_size + 2

print(f'인코딩 테스트 - 전: {sentences[19]}')
encodingList = tokenizer.encode(sentences[19])
print(f'인코딩 테스트 - 후: {encodingList}')
# for tok in encodingList:
#     print(tokenizer.subwords[tok-1])

answer_lengths = [len(s) for s in abs]
avg_abs = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
max_abs = max(answer_lengths) if answer_lengths else 0

question_lengths = [len(s) for s in sentences]
avg_sentences = sum(question_lengths) / len(question_lengths) if question_lengths else 0
max_sentences = max(question_lengths) if question_lengths else 0

print(f"abs - 평균 길이: {avg_abs:.2f}, 최대 길이: {max_abs}")
print(f"sentences - 평균 길이: {avg_sentences:.2f}, 최대 길이: {max_sentences}")
input('enter')
MAX_LENGTH = 6990
def tokenizer_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        
        tokenized_inputs.append(sentence1)
        tokenized_outputs.append(sentence2)

    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post'
    )
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post'
    )

    return tokenized_inputs, tokenized_outputs

sentences, abs = tokenizer_and_filter(sentences, abs)
"""
print(sentences[19])
print(abs[19])
"""

BATCH_SIZE = 128
BUFFER_SIZE = 20000

dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs':sentences,
        'dec_inputs':abs[:, :-1]
    },
    {
        'outputs':abs[:, 1:]
    }
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

def transformer(vocab_size, num_layers, dff,
                d_model, num_heads, dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")
  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask, output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)
  enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
      d_model=d_model, num_heads=num_heads, dropout=dropout,
  )(inputs=[inputs, enc_padding_mask]) 

  dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
      d_model=d_model, num_heads=num_heads, dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

class PositionalEncoding(tf.keras.layers.Layer):
  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)

    sines = tf.math.sin(angle_rads[:, 0::2])

    cosines = tf.math.cos(angle_rads[:, 1::2])

    angle_rads = np.zeros(angle_rads.shape)
    angle_rads[:, 0::2] = sines
    angle_rads[:, 1::2] = cosines
    pos_encoding = tf.constant(angle_rads)
    pos_encoding = pos_encoding[tf.newaxis, ...]

    print(pos_encoding.shape)
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)
def encoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")

  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
        dropout=dropout, name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs, 'key': inputs, 'value': inputs,
          'mask': padding_mask
      })

  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)

  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    outputs = self.dense(concat_attention)

    return outputs

def scaled_dot_product_attention(query, key, value, mask):
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  if mask is not None:
    logits += (mask * -1e9)

  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output, attention_weights

def decoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
        dropout=dropout, name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")

  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs, 'key': inputs, 'value': inputs,
          'mask': look_ahead_mask
      })

  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1, 'key': enc_outputs, 'value': enc_outputs,
          'mask': padding_mask
      })

  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

  outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)

  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

tf.keras.backend.clear_session()

D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1

model = transformer(
  vocab_size=VOCAB_SIZE,
  num_layers=NUM_LAYERS,
  dff=DFF,
  d_model=D_MODEL,
  num_heads=NUM_HEADS,
  dropout=DROPOUT
)
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(
  learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

def acuuracy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[acuuracy])

EPOCHS = 50
model.fit(dataset, epochs=EPOCHS)

def preprocess_sentence(sentence):
  sentence = sentence.str.replace(r'\S+@\S+\.\S+', '', regex=True)
  sentence = sentence.str.replace(r'\s+', ' ', regex=True).str.strip()
  sentence = sentence.strip()
  return sentence

def evaluate(sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)
def predict(sentence):
  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('Bot : {}'.format(predicted_sentence))

  return predicted_sentence

# output = predict("독감환자 외래환자, 1천명 당 71.8명 지속 증가 이성훈 여수시가 최근 이례적으로 A형과 B형 독감이 동시에 유행하며 많은 환자가 발생하고 있는 가운데 고위험군인 어르신들의 건강관리에 나선다. 교육은 독감의 증상부터 예방접종의 중요성, 일상생활에서의 예방수칙 등을 주제로 진행된다. 질병관리본부의 발표에 따르면 지난해 12월 말 기준 독감 환자는 외래환자 1000명 당 71.8명으로 계속해서 증가 추세를 보이고 있다. 흔히‘독감’이라고 하는 인플루엔자는 인플루엔자 바이러스에 의해 감염되며, 폐렴 등 합병증이 유발될 수 있어서 주의가 필요하다. 인플루엔자 바이러스에 감염되면 1~4일 후에 발열, 두통, 근육통, 콧물, 기침 등의 증상이 나타난다. 전문가들은 일상생활에서 예방수칙 준수도 강조한다. 시 관계자는“경로당과 어린이집, 요양시설 등을 대상으로 예방수칙 준수 교육을 집중적으로 추진할 계획”이라며 “시민들께서도 예방접종과 올바른 손 씻기, 기침예절 준수 등에 적극 협조해 주시기 바란다”고 말했다.")
# output = predict("고용창출·경제 활성화 기대 이성훈 지역 업체인 이에스바이오 농업회사법인주식회사(이하 이에스바이오·대표 강기연)와 광양시는 지난 19일 시민접견실에서 투자협약을 체결했다. 협약식에는 정현복 시장과 강기연 이에스바이오 대표를 비롯한 관계자 10여명이 참석했다. 이번에 신규 투자를 결정한 이에스바이오는 상토 및 유기질 비료 생산 전문 업체로 분말형상토, 압착매트형상토, 양액배지, 유기질 비료 등 농작물의 종류에 맞는 제품을 생산하는 기술을 보유하고 있으며, 생산품을 일본·중국 등지로 수출을 계획하고 있다. 이번 협약으로 이에스바이오는 광양항 동측배후단지 2만9113㎡의 부지에 41억원을 투자해 내년까지 상토 및 유기질 비료 제조 공장을 설립하게 된다. 또 광양항 동측배후단지에 입주해 물류비 절감과 수입 원자재에 대한 원가 절감 등을 통한 수익을 극대화해 나갈 예정이다. 시는 이번 투자유치를 통해 관리직·생산직을 비롯한 40여명이 양질의 새로운 일자리를 갖게 되고, 지역 경제 활성화에 도움이 될 것으로 기대하고 있다. 정현복 광양시장은“이번 투자 협약이 우리시가 기업하기 좋은 도시로 한걸음 더 나아가는 계기가 될 것이다”며“시는 앞으로도 경쟁력 있는 기업이 성공할 수 있도록 최선을 다해 돕겠다”고 말했다. 강기연 대표는“광양시와 투자협약을 맺게 되어 기쁘고 지역 발전에 대한 책임감이 더욱더 커진다”며“이번 협약으로 지역 발전에 기여하고 광양항 활성화를 위해 더욱더 최선을 다하겠다”고 소감을 밝혔다.")
# output = predict("[옥천]자유한국당 박덕흠 의원(보은·옥천·영동·괴산군)은 환경부에서 지정폐기물을 수거해 직접처리를 의무화하는 '폐기물 관리법' 일부개정 법률안을 대표 발의했다고 지난 18일 밝혔다. 이 대표발의한 개정안의 내용은 환경오염유발 및 위해성 높은 지정폐기물의 환경부 직접수거처리, 조례를 통한 폐기물처리시설의 설치제한, 폐기물처리시설 주변지역에 대한 주민지원사업시행 등이다. 현행 법령에서 폐유 폐산 등 주변환경을 오염시킬 수 있거나 인체에 위해를 줄 수 있는 지정폐기물은 엄격하게 관리하도록 규정하고 있지만 폐기물의 일부만 소각되고 나머지는 전국 각지의 바다, 산, 땅속에 버려지고 있어 국가차원의 관리가 필요하다는 의견이 대두되어 왔다. 폐기물이 소각되는 경우에도 폐기물의 발생지는 대부분 수도권인 반면 폐기물 처리시설은 수도권이외 농어촌지역에 설치되어 해당지자체와 지역주민들의 불만이 많아 관련법령의 개정이 시급한 상황이다. 최근 충북지역 괴산군과 옥천군에 폐기물처리업체가 들어설 준비를 하고 있어 해당 지역주민들의 행복권과 건강이 위협받는 심각한 상황으로 지역주민들이 삭발식 등 시위를 통해 격렬히 반발하고 있다. 박 의원은 ""최근 괴산군과 옥천군에 폐기물처리시설 업체가 들어올 것이라는 지역주민들의 얘기를 듣고 법안을 발의하게 되었다""며 ""앞으로 법안이 통과되면 폐기물처리에 있어 힘없는 지자체의 힘겨운 싸움이 아니라 환경부 등 정부차원에서 체계적으로 관리하고 지역주민들에 대한 지원금도 상당한 금액이 될 수 있도록 법안통과에 힘 쓰겠다""고 법안통과 의지를 밝혔다.")
# output = predict("노조 파업 투표…찬성 94% 대전 시내버스노조가 10일 파업 찬반투표를 한 결과 찬성이 압도적인 우위를 점하면서 시내버스 파업 현실화가 한 걸음 앞당겨졌다. 대전 시내버스노조에 따르면 노조는 이날 진행한 파업 찬반투표 결과, 찬성이 94%를 차지해 오는 17일 버스 파업 돌입을 예고했다. 노조와 대전버스운송사업조합은 그 동안 5차례에 걸쳐 임금 및 단체협약을 진행했지만 합의점을 찾지 못했다. 노조는 내년부터 주 52시간 근무제가 시행됨에 따라 근무일수 24일 보장과 7.67%의 임금 인상을 요구했지만 사측은 근무일수 23일 보장에 임금 2.0% 인상을 주장하며 견해 차이를 좁히지 못했다. 이에 따라 노조는 지난 1일 충남지방노동위원회에 노동쟁의 조정 신청을 했고, 15일간 2차례에 걸쳐 조정회의가 진행된다. 이번 노조의 파업결정으로 남은 조정 기간 동안 사측과 합의점을 찾지 못한다면 예고된 17일 대전 시내버스 파업은 현실화된다. 대전시내의 버스 파업 논란이 불거진 건 이번만이 아니다. 노조는 2012년 당시 국회의 택시법 발의와 관련해 버스운행 중단을 선언했다가 2시간만에 철회했다. 지난 4월에도 노조는 충남지방노동위원회의 노동쟁의 신청을 했지만 파업 찬반투표까지 치닫지는 않았다. 대전 시내버스 파업이 현실화 된다면 2007년 이후 12년 만이다. 대전시도 만일의 사태에 대비해 전세버스 200대와 관용버스 34대를 투입할 계획이다. 정상운행 되는 버스 411대 까지 합해진다면 파업기간 동안 운행되는 버스는 총 645대로 이는 평일 대전시 버스 정상 운행률의 66.8%, 주말에는 78.9% 수준이다. 도시철도도 하루 240회에서 290회로 늘려 운행에 시민들의 피해를 최소하겠다는 계획이다. 그러나 파업에 돌입할 경우, 그 기간이 길어진다면 시민들의 피해는 불가피 할 것으로 보인다. 2007년 대전시내버스 파업 당시에도 사측과의 협상이 난항을 겪어 파업기간이 12일까지 길어진 전례가 있다. 대전버스노조 관계자는 ""파업 여부는 사측에 달렸다""며 ""12일에 사측과 1차 조정회의가 있고 쟁의조정기간도 남아 있는 만큼 노조의 요구안을 사측이 수용한다면 파업은 철회될 수 있다""고 말했다. 이정훈 기자 수습 김기운 기자")