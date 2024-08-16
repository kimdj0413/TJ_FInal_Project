import requests
from bs4 import BeautifulSoup
import re
import time

query = '삼성'
page = 0
title_list = []

# User-Agent 설정
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'ko-KR,ko;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

for i in range(0, 3):
    page += 1
    url = f'https://search.daum.net/search?w=news&nil_search=btn&DA=STC&enc=utf8&cluster=y&cluster_page=1&q={query}&p={page}&sort=recency'
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        title_elements = soup.find_all(class_='tit-g clamp-g')
        for title_element in title_elements:
            title_list.append(title_element.get_text())
    else:
        print("웹 페이지를 가져오는 데 실패했습니다.")

print(title_list)
print(len(title_list))