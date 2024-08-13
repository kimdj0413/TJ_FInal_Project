import requests
from bs4 import BeautifulSoup
import csv
import time

# CSV 파일 생성 및 헤더 작성
with open('naver_kin_data.csv', mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(["Page", "Question Title", "Question Content", "Answer Content"])

    for page in range(1, 100):  # 1부터 99페이지까지
        url = f"https://kin.naver.com/qna/list.naver?dirId=40102&page={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 각 질문 항목을 찾기
        questions = soup.select('ul.basic1 li')

        for question in questions:
            title_tag = question.select_one('a._nclicks:kin.txt._title')
            if title_tag:
                question_title = title_tag.text.strip()
                question_link = "https://kin.naver.com" + title_tag['href']
                
                # 질문 페이지에 접근
                question_response = requests.get(question_link)
                question_soup = BeautifulSoup(question_response.text, 'html.parser')
                
                # 질문 내용
                question_content_tag = question_soup.select_one('div._endContentsText')
                question_content = question_content_tag.text.strip() if question_content_tag else "No content"
                
                # 답변 내용 (최초 답변만 크롤링)
                answer_content_tag = question_soup.select_one('div._endContentsText')
                answer_content = answer_content_tag.text.strip() if answer_content_tag else "No answer"
                
                # CSV 파일에 작성
                writer.writerow([page, question_title, question_content, answer_content])
                
                # 과도한 요청을 피하기 위해 잠시 대기
                time.sleep(1)

        print(f"Page {page} completed")

print("Crawling completed.")
