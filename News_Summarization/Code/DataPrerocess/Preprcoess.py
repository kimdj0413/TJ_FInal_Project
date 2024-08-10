import pandas as pd
import unicodedata

df = pd.read_csv('D:/TJ_FInal_Project/News_Summarization/Data/문서요약 텍스트/Preprocess/train.csv')
df = df.iloc[:, 1:]
df = df.dropna()
df = df[df['sentence'].str.len() > 10]
df = df[df['abs'].str.len() > 10]
df['sentence'] = df['sentence'].str.replace(r'\S+@\S+\.\S+', '', regex=True)
df['sentence'] = df['sentence'].str.replace(r'\s+', ' ', regex=True).str.strip()
df['abs'] = df['abs'].str.replace(r'\s+', ' ', regex=True).str.strip()
df['sentence'] = df['sentence'].str.replace('\n', '', regex=False)
df['abs'] = df['abs'].str.replace('\n', '', regex=False)
print(df)

preprocess_sentence = []
preprocess_abs = []
for sentence in df['sentence']:
    preprocess_sentence.append(sentence)
for sentence in df['abs']:
    preprocess_abs.append(sentence)

lengths = [len(x) for x in preprocess_sentence]
maxLength = max(lengths)
print(maxLength)
df = pd.DataFrame({'sentence':preprocess_sentence, 'abs':preprocess_abs})
df.to_csv('D:/TJ_FInal_Project/News_Summarization/Data/문서요약 텍스트/Preprocess/train_preprocess.csv', index=False)
