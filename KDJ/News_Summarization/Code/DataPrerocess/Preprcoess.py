import pandas as pd
import unicodedata
import re
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

"""
######################
##      1차처리     ##
######################
df = pd.read_csv('D:/TJ_FInal_Project/News_Summarization/Data/문서요약 텍스트/Preprocess/train.csv')
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
df.to_csv('D:/TJ_FInal_Project/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess.csv', index=False)

##########################
##      2차 전처리      ##
##########################

df = pd.read_csv('D:/TJ_FInal_Project/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess.csv')

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

df.to_csv('D:/TJ_FInal_Project/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess.csv',index=False)

"""
data = pd.read_csv('D:/TJ_FInal_Project/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess.csv')
train_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
sentences = []
for sentence in train_data['sentence']:
    sentences.append(sentence)
abs = []
for sentence in train_data['abs']:
    abs.append(sentence)

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    sentences + abs, target_vocab_size = 2**13
)
tokenizer.save_to_file('D:/TJ_FInal_Project/News_Summarization/Data/문서요약 텍스트/Preprocess/tokenizer')
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('D:/TJ_FInal_Project/News_Summarization/Data/문서요약 텍스트/Preprocess/tokenizer')
print(tokenizer.subwords[:100])

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size+1]
VOCAB_SIZE = tokenizer.vocab_size + 2

print(f'인코딩 테스트 - 전: {sentences[19]}')
encodingList = tokenizer.encode(sentences[19])
print(f'인코딩 테스트 - 후: {encodingList}')
for tok in encodingList:
    print(tokenizer.subwords[tok-1])

answer_lengths = [len(s) for s in abs]
avg_abs = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
max_abs = max(answer_lengths) if answer_lengths else 0

question_lengths = [len(s) for s in sentences]
avg_sentences = sum(question_lengths) / len(question_lengths) if question_lengths else 0
max_sentences = max(question_lengths) if question_lengths else 0

print(f"abs - 평균 길이: {avg_abs:.2f}, 최대 길이: {max_abs}")
print(f"sentences - 평균 길이: {avg_sentences:.2f}, 최대 길이: {max_sentences}")
input('enter')