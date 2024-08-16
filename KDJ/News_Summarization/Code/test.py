import pandas as pd
import re

data = pd.read_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/finalPreprocess.csv')

data = data.iloc[:1000]
data.to_csv('D:/TJ_FInal_Project/KDJ/News_Summarization/Data/문서요약 텍스트/Preprocess/finalPreprocessTemp.csv')