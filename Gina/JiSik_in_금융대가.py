
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

# Chrome 드라이버의 경로를 설정합니다.
chrome_driver_path = 'C:/TJ_FInal_Project/Gina/chromedriver.exe'  # 실제 경로로 변경하세요

# Chrome 드라이버 설정
chrome_options = Options()
chrome_options.add_argument("--headless")  # 헤드리스 모드로 실행 (UI가 필요 없는 경우)

# 드라이버 객체 생성
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

def scrape_page_content(driver, link_xpath):
    # 링크 클릭
    link_element = driver.find_element(By.XPATH, link_xpath)
    link_element.click()
    
    time.sleep(2)  # 페이지 로딩 대기

    # 본문 내용 추출
    try:
        content = driver.find_element(By.CSS_SELECTOR, '#content > div.endContentLeft._endContentLeft > div.contentArea._contentWrap > div.questionDetail').text
        contentlist = driver.find_element(By.CSS_SELECTOR, '#content > div.endContentLeft._endContentLeft > div._contentBox.contentBox.contentBox--headerAnswerContent').text
        
        # contentlist = driver.find_element(By.CSS_SELECTOR, '#SE-38ef31f4-6f8b-4358-93ca-f65409826e2e').text
        # parent_element = driver.find_element(By.CSS_SELECTOR, '#content > div.endContentLeft._endContentLeft > div._contentBox.contentBox.contentBox--headerAnswerContent')
        # contentlist = parent_element.find_element(By.CSS_SELECTOR, '#SE-f2107a6c-a903-48e8-84f1-4ed6e25e1eae').text
        
        print(f'질문 내용: {content[:50]}')  # 일부만 출력
        print(f'답변 내용: {contentlist[:50]}')
    except Exception as e:
        print(f'본문 크롤링 중 오류 발생: {e}')
        content = None
        contentlist = None
        
    # 다시 이전 페이지로 돌아가기
    driver.back()
    time.sleep(2)  # 페이지 로딩 대기

    return content, contentlist

def main():
    base_url = "https://kin.naver.com/userinfo/answerList.naver?u=OmgAyX6ncx9LcS8YNEj8kNAE8NgxJHHz7%2B%2BsHxCcLQA%3D&page="
    
    contents = []
    
    for page in range(1, 100):  # 1부터 99페이지까지 순회
        current_url = base_url + str(page)
        print(f'현재 페이지: {page} ({current_url}) 크롤링 중...')
        driver.get(current_url)
        time.sleep(2)  # 페이지 로딩 대기
        
        for i in range(1, 21):  # 각 페이지에서 20개의 목록을 순회
            xpath = f'//*[@id="au_board_list"]/tr[{i}]/td[1]/a'  # 각 목록의 XPath 생성
            print(f'{i}번째 글 크롤링 중...')
            content, contentlist = scrape_page_content(driver, xpath)
            if content and contentlist:  # 질문과 답변이 모두 있을 때만 추가
                contents.append({"Question": content, "Answer": contentlist})
        
    driver.quit()  # 브라우저 닫기
    print("크롤링 완료")
    
    # 크롤링한 내용을 데이터프레임으로 변환
    df = pd.DataFrame(contents)

    # CSV 파일로 저장
    csv_file_path = './Gina/jisik_in_금융대가의꿈.csv'
    df.to_csv(csv_file_path, mode='a', index=False, encoding='utf-8-sig')
    
    print(f"크롤링한 내용을 '{csv_file_path}' 파일로 저장했습니다.")

if __name__ == "__main__":
    main()