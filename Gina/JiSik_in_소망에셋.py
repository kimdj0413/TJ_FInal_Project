import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup 
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
    """주어진 XPath를 사용하여 본문 내용을 크롤링합니다."""
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    # 링크 클릭
    link_element = driver.find_element(By.XPATH, link_xpath)
    link_element.click()
    
    time.sleep(2)  # 페이지 로딩 대기 (필요에 따라 조정 가능)

    # 본문 내용 추출 (여기서도 CSS 선택자는 실제 페이지에 맞게 조정 필요)
    try:
        content = driver.find_element(By.CSS_SELECTOR, '#content > div.endContentLeft._endContentLeft > div.contentArea._contentWrap > div.questionDetail').text
        print(f'질문 내용: {content[:50]}')  # 일부만 출력
        contentlist = driver.find_element(By.CSS_SELECTOR, '#answer_1 > div.answerDetail._endContents._endContentsText > div > div').text
        
        # content_ = soup.select_one(".se-main-container")
        # contentlist = soup.find(class_="se-module se-module-text").text
        # contentlist = driver.find_element(By.XPATH, '//*[@id="SE-ac59bf6e-5711-4f18-b606-f551e55c1c5c"]/div')
        # text = contentlist.get_text()
        print(f'답변 내용:{contentlist}')
    except Exception as e:
        print(f'본문 크롤링 중 오류 발생: {e}')
        content = None
        
    # 다시 이전 페이지로 돌아가기
    driver.back()
    time.sleep(2)  # 페이지 로딩 대기

    return content, contentlist

def main():
    base_url = "https://kin.naver.com/userinfo/answerList.naver?u=%2B15vL%2B%2FjcNIEWf9812DrQRT0PsnVuSW4drAvhSWSPM0%3D&page=1"
    driver.get(base_url)  # 시작 페이지로 이동
    
    contents = []
    for i in range(1, 21):  # 20개의 목록을 순회
        xpath = f'//*[@id="au_board_list"]/tr[{i}]/td[1]/a'  # 각 목록의 XPath 생성
        print(f'{i}번째 글 크롤링 중...')
        content , contentlist = scrape_page_content(driver, xpath)
        if content:
            contents.append(content)
    
    driver.quit()  # 브라우저 닫기
    
    # 크롤링한 내용을 데이터프레임으로 변환
    df = pd.DataFrame(contents, contentlist, columns=["Question" , "Answer"])

    # CSV 파일로 저장
    csv_file_path = './Gina/crawled_content.csv'  # 저장할 파일 이름 및 경로
    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
    
    print(f"크롤링한 내용을 '{csv_file_path}' 파일로 저장했습니다.")

if __name__ == "__main__":
    main()
