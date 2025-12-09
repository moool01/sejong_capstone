# 우리역사넷 크롤링
import requests
from bs4 import BeautifulSoup
from pathlib import Path

url = "https://contents.history.go.kr/mobile/kc/view.do?levelId=kc_n305800"
headers = {"User-Agent": "Mozilla/5.0"}

res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, "lxml")

# 본문 섹션 내 모든 문단 <p> 추출
paragraphs = [p.get_text(" ", strip=True) for p in soup.select("div.tx p") if p.get_text(strip=True)]

# txt 파일로 저장
output_path = Path("우리역사넷crawling.txt")
with output_path.open("w", encoding="utf-8") as f:
    for para in paragraphs:
        f.write(para + "\n\n")   # 문단 사이에 빈 줄 넣기

print(f"총 {len(paragraphs)}개 문단 저장 완료 → {output_path.resolve()}")
