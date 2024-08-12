import pandas as pd
import unicodedata
import re
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import BertTokenizer
from tqdm import tqdm

"""
######################
##      1차처리     ##
######################
df = pd.read_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/train.csv')
##      title 열 날리기
df = df.iloc[:, 1:]

##      난수 날리기
df = df.dropna()

##      길이가 10미만인 행 날리기
df = df[df['sentence'].str.len() > 10]
df = df[df['abs'].str.len() > 10]

##      정규표현식
df['sentence'] = df['sentence'].str.replace(r'\S+@\S+\.\S+', '', regex=True)        # 이메일
df['sentence'] = df['sentence'].str.replace(r'\s+', ' ', regex=True).str.strip()    # 연속된 공백을 하나로
df['abs'] = df['abs'].str.replace(r'\s+', ' ', regex=True).str.strip()
df['sentence'] = df['sentence'].str.replace('\n', '', regex=False)                  # 개행 삭제
df['abs'] = df['abs'].str.replace('\n', '', regex=False)
print(df)

##      새로운 데이터프레임 생성
preprocess_sentence = []
preprocess_abs = []
for sentence in df['sentence']:
    preprocess_sentence.append(sentence)
for sentence in df['abs']:
    preprocess_abs.append(sentence)

##      길이 분포 확인
lengths = [len(x) for x in preprocess_sentence]
maxLength = max(lengths)
print(maxLength)
df = pd.DataFrame({'sentence':preprocess_sentence, 'abs':preprocess_abs})
df.to_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess.csv', index=False)

##########################
##      2차 전처리      ##
##########################

df = pd.read_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess.csv')

##      정규표현식
def replace_special_characters(series):     # 영어,한자, 숫자, 한글, 공백, ,!?'"~ = 공백 으로 대치
    return series.apply(lambda x: re.sub(r'[^가-힣a-zA-Z0-9\u4e00-\u9fff\s.,!?\'\"~]', ' ', x))
df['sentence'] = replace_special_characters(df['sentence'])
df['sentence'] = df['sentence'].str.replace(r'\s+', ' ', regex=True).str.strip()

##      10 < 문자열 길이 < 1914
df = df[df['sentence'].str.len() > 10]
df = df[df['sentence'].str.len() < 1912]

##      길이 확인하기
df['length'] = df['sentence'].apply(len)

max_length = df['length'].max()
min_length = df['length'].min()
average_length = df['length'].mean()

print(f'데이터프레임 길이: {len(df)}')
print(f"최대 길이: {max_length}")
print(f"최소 길이: {min_length}")
print(f"평균 길이: {average_length:.2f}")

plt.figure(figsize=(10, 6))
plt.hist(df['length'], bins=range(min_length, max_length + 2), alpha=0.7, color='blue', edgecolor='black')
plt.title('Length Distribution of Column')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.xticks(range(min_length, max_length + 1))
plt.grid(axis='y', alpha=0.75)
plt.show()

df.to_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess.csv',index=False)
"""
######################
##      토큰화      ##
######################
##      tensorflow
# data = pd.read_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess.csv')
# data = data.iloc[:10000]
# train_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
# train_data.to_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess_test.csv')
train_data = pd.read_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess_test.csv')
sentences = []
for sentence in train_data['sentence']:
    sentences.append(sentence)
abs = []
for sentence in train_data['abs']:
    abs.append(sentence)

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    sentences + abs, target_vocab_size = 2**20
)
tokenizer.save_to_file('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/tokenizer')
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/tokenizer')
print(tokenizer.subwords[:100])
print(f'토큰 크기: {tokenizer.vocab_size}')
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size+1]
VOCAB_SIZE = tokenizer.vocab_size + 2

print(f'인코딩 테스트 - 전: {sentences[19]}')
encodingList = tokenizer.encode(sentences[19])
print(f'인코딩 테스트 - 후: {encodingList}')
# for tok in encodingList:
#     print(tokenizer.subwords[tok-1])
# subwords 출력
for tok in encodingList:
    # tok 값이 0보다 크고 subwords의 길이 범위 내에 있는지 확인
    if tok > 0 and tok - 1 < len(tokenizer.subwords):
        print(tokenizer.subwords[tok - 1])
    else:
        print(f"토큰 {tok}는 유효한 인덱스가 아닙니다.")
print('dd')
answer_lengths = [len(s) for s in abs]
avg_abs = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
max_abs = max(answer_lengths) if answer_lengths else 0

question_lengths = [len(s) for s in sentences]
avg_sentences = sum(question_lengths) / len(question_lengths) if question_lengths else 0
max_sentences = max(question_lengths) if question_lengths else 0

print(f"abs - 평균 길이: {avg_abs:.2f}, 최대 길이: {max_abs}")
print(f"sentences - 평균 길이: {avg_sentences:.2f}, 최대 길이: {max_sentences}")

# ##      KoBERT
# tokenizer = BertTokenizer.from_pretrained('monologg/kobert')

# data = pd.read_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess.csv')
# train_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# sentences = train_data['sentence'].tolist()
# abs = train_data['abs'].tolist()

# # tokens = []
# # for sentence, summary in tqdm(zip(sentences, abs), total=len(sentences), desc="토큰화 진행 중"):
# #     tokens.append(tokenizer.tokenize(sentence + summary))

# # tokenizer.save_pretrained('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/tokenizer')

# tokenizer = BertTokenizer.from_pretrained('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/tokenizer')
# tokens = tokenizer.tokenize("국내 클라우드 서비스 제공업체 네이버 비즈니스 플랫폼은 글로벌 상호연결 및 데이터센터 전문기업 에퀴닉스 Equinix 와 함께 글로벌 클라우드 비즈니스 확장에 나선다고 21일 밝혔다. NBP는 자사 클라우드 서비스 품질 및 네트워크 가용성 향상을 위해 '플랫폼 에퀴닉스 Platform Equinix '를 도입했다. NBP의 고객사는 에퀴닉스 인터넷 익스체인지 Equinix Internet Exchange 를 통해 전 세계 어디서든 1800개 이상의 네트워크 서비스 제공업체 NSP 에 접근할 수 있다. 최근 싱가포르, 미국, 홍콩, 일본, 독일 등으로 서비스를 확대하고 있는 NBP는 현재 글로벌 고객에게 25개 카테고리 중 124개의 클라우드 서비스를 제공하고 있다.")
# print(tokens)
# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print("토큰 ID:", token_ids)
