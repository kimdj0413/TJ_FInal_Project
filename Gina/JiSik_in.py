import pandas as pd
import numpy as np
import platform
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup 
from urllib.request import urlopen
import urllib
import time

tmp1 = 'https://search.naver.com/search.naver?where=kin'
# 네이버 지식인 Url 뒤에 kin부분만 바꾸면 뉴스 블로그 카페 다 가능
html = tmp1 + '&sm=tab_jum&ie=utf8&query={key_word}&kin_start={num}' 
# 네이버 지식인 페이지수 결정 Url

response = urlopen(html.format(num = 20, key_word = urllib.parse.quote('국내주식')))
#urlib.parse.quite를 사용하는 이유는 Url에 한글을 섞으면 오류가 발생하는데 그 오류를 방지하기 위한 함수이다.

soup = BeautifulSoup(response, "html.parser")

tmp = soup.find_all('dl')

tmp_list = []
for line in tmp:
    tmp_list.append(line.text)  
print(tmp_list)


#타임바를 시각적으로 표현 할 수 있는 패키지
from tqdm import tqdm  

stock_market_ko = []
for n in tqdm(range(1, 1000, 10)): # range(1,1000),mininterval = 10
    response = urlopen(html.format(num=n, key_word=urllib.parse.quote('국내주식')))
   
    soup = BeautifulSoup(response, "html.parser")

    tmp = soup.find_all('dl')

    for line in tmp:
        stock_market_ko.append(line.text)
        
    time.sleep(0.5)

print(stock_market_ko[:10])  # 처음 몇 개의 데이터를 출력하여 크롤링이 제대로 이루어졌는지 확인

# 자연어 처리에 필요한 패키지를  불러오고 크롤링한 리스트형태의 구조를 스트링 구조로 바꿔준다.
import nltk
import re
from konlpy.tag import Okt; t = Okt() 

df_stock_market_ko = str(stock_market_ko)


# 특수문자 제거 하기
def cleanText(readData):
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
    return text

# morphs함수로 형태소 추출
tokens_ko = t.morphs(cleanText(df_stock_market_ko))
print(tokens_ko)  # 형태소 분석 후 결과 확인

# 불용어
stop_words = ['가','요','답변','을','수','에','질문','제','를','이','도',
              '좋','는','로','으로','것','은','다','니다','대','들',
              '들','데','의','때','겠','고','게','네요','한','일','할',
              '하는','주','려고','인데','거','좀','는데',
              '하나','이상','뭐','까','있는','잘','습니다','다면','했','주려',
              '지','있','못','후','중','줄','과','어떤','기본',
              '단어','선물해','라고','중요한','합','가요','보이','네','무지','합니다','ㅠㅠ','에서']

# 불용어 처리하고 많이 언급된 상위 50개 
tokens_ko = [each_word for each_word in tokens_ko if each_word not in stop_words]

ko = nltk.Text(tokens_ko, name='국내주식')
ko.vocab().most_common(50)

data = ko.vocab().most_common(50)
print(data)  # 빈 데이터가 출력되는지 확인


'''
# 시각화
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(15,6))
ko.plot(50) 
plt.show()
'''

# 워드클라우드 시각화
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

data = ko.vocab().most_common(50)

if len(data) > 0:
    wordcloud = WordCloud(font_path='C:/Windows/Fonts/NanumBarunGothic.ttf',
                          relative_scaling=0.2,
                          stopwords=STOPWORDS,
                          background_color='white').generate_from_frequencies(dict(data))
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
else:
    print("No words available to generate word cloud.")
    wordcloud = None  # 워드클라우드가 없음을 명시적으로 표현

# Word2Vec 모델링 부분에서 wordcloud가 있는 경우에만 시각화
if wordcloud:
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()



#  gensim 패키지를 통해 긍부정을 분석
import gensim
from gensim.models import word2vec

Okt_ = Okt()
results = []
lines = stock_market_ko


for line in lines:
    malist = Okt_.pos(line, norm=True, stem=True)
    r= []
    
    for word in malist:
        if not word[1] in ["Josa", "Eomi", "Punctuation"]:
            r.append(word[0])
            
    r1 = (" ".join(r)).strip()
    results.append(r1)
    print(r1)



# 긍정부정 분석 결과를 저장하여 word2vec로 긍부정 알아보기 
data_file = '국내주식.data'
with open(data_file, 'w', encoding='utf-8') as fp:
    fp.write("\n".join(results))

data = word2vec.LineSentence(data_file)
model = word2vec.Word2Vec(data, vector_size=200, window=10, hs=1, min_count=2, sg=1)  # size -> vector_size로 변경
model.save('국내주식.model')

model = word2vec.Word2Vec.load('국내주식.model')
similar_words = model.wv.most_similar(positive=['상승'], negative=['하락'])  # wv 속성으로 접근해야 함
print(similar_words)

