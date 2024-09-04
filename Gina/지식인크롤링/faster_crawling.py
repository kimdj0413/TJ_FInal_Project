import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

#주소 가져오기 
preset = 'https://kin.naver.com'
url_list = []

for i in range(1, 100):
  url = f'https://kin.naver.com/userinfo/answerList.naver?u=P%2BJ0EbNqgtwj0cD5Ko9RY66xb4PYWcAn9aVqkXa5TB4%3D&page={i}'
  response = requests.get(url)

  if response.status_code == 200:
      soup = BeautifulSoup(response.text, 'html.parser')
      title_elements = soup.find_all('td', class_='title')
      
      for title in title_elements:
          link_element = title.find('a')
          if link_element and 'href' in link_element.attrs:
              href = link_element['href']
              url_list.append(preset + href)
          else:
              print("링크가 없습니다.")
  else:
      print(f"페이지 요청 실패: {response.status_code}")

url_list = [s.replace("mydetail", "detail") for s in url_list]
print(url_list[5])
print(f'가져온 url 갯수 : {len(url_list)}')

time.sleep(2)

# 질문가져오기
question_list = []

for idx, url in enumerate(url_list):
    page_num = (idx // 20) + 1  # 페이지 번호 계산 (1페이지당 10개의 질문이라고 가정)
    print(f"{page_num} 페이지에서 {idx + 1}번째 질문을 크롤링 중입니다.")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    title_class = 'endTitleSection'
    elements = soup.find(class_=title_class)
    if elements:
        title = elements.get_text(strip=True)
        title = title[2:]
    else:
        title = ""  # 제목을 찾지 못한 경우 처리

    qustion_class = 'questionDetail'
    try:
        elements = soup.find(class_=qustion_class)
        if elements:
            question = elements.get_text(strip=True)
        else:
            question = ""  # 질문 세부사항을 찾지 못한 경우 처리
    except AttributeError:
        question = ""  # 예기치 않은 AttributeError 처리

    question_list.append(title + " " + question)

print(question_list[5])
print(f'질문 길이 : {len(question_list)}')

time.sleep(2)

# 답변가져오기
answer_list = []

for idx, url in enumerate(url_list):
  page_num = (idx // 20) + 1  # 페이지 번호 계산
  print(f"{page_num} 페이지에서 {idx + 1}번째 답변을 크롤링 중입니다.")
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')

  answer_class = 'answerArea _contentWrap _answer'
  elements = soup.find(class_=answer_class)
  
  answer_class = 'answerDetail _endContents _endContentsText'
  answer = soup.find(class_=answer_class)
  try:
    final_answer = answer.get_text(strip=True)
    answer_list.append(final_answer)
  except AttributeError:
    final_answer = ""
    answer_list.append(final_answer)

print(answer_list[0:5])
print(f'답변 갯수 : {len(answer_list)}')

df = pd.DataFrame({
  'Question' : question_list,
  'Answer' : answer_list
})

df.to_csv('./코인선물거래전문가.csv', index=False)
print("csv파일이 성공적으로 저장되었습니다.")
