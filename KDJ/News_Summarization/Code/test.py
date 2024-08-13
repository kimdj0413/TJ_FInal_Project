import pandas as pd

valid_data = pd.read_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/valid.csv')
valid_data = valid_data.iloc[:5]
valid_data = valid_data.iloc[:,1:]
valid_data.to_csv('KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/valid_test.csv', index=False)
sentence = valid_data['sentence']
for i in range(0,5):
    print(sentence[i])