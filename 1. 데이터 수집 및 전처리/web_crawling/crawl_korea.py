# 한국민족대백과사전 크롤링 - 입력값: 세종, depth=1로해서 세종 관련 인물,파일도 같이 크롤링
import os
import time
import re
from urllib.parse import quote, urljoin
import requests
from bs4 import BeautifulSoup

BASE = "https://encykorea.aks.ac.kr"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

# -----------------------------
# 유틸
# -----------------------------
def slugify(filename: str) -> str:
    """파일명 안전하게"""
    filename = re.sub(r"[\\/:*?\"<>|]", "_", filename)
    filename = filename.strip().rstrip(".")
    return filename or "output"

def get_json(url: str):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def get_html(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def clean_text(node) -> str:
    """불필요 요소 제거 후 텍스트 정리"""
    if node is None:
        return ""
    # 각주/주석 상단 표기 제거
    for sup in node.select("sup"):
        sup.decompose()
    # 토글 버튼, 아이콘 등 제거
    for b in node.select(".btn-detail-remark, .btn-detail-line, .tooltip-basic, .tooltip-contents"):
        # 버튼 안 텍스트는 살리고 태그만 제거
        b.unwrap()
    txt = node.get_text(" ", strip=True)
    # 공백 정리
    return re.sub(r"\s+", " ", txt).strip()

def extract_paragraphs(container) -> list[str]:
    """<p> 단위 문단 분리. <p>가 없으면 전체를 1문단으로 반환."""
    if container is None:
        return []
    paras = []
    ps = container.select("p")
    if ps:
        for p in ps:
            t = clean_text(p)
            if t:
                paras.append(t)
    else:
        t = clean_text(container)
        if t:
            paras = [t]
    return paras

# -----------------------------
# 크롤링 코어
# -----------------------------
def first_article_eid(keyword: str) -> str:
    """자동완성 API에서 첫 번째 항목 eid 가져오기"""
    url = f"{BASE}/Article/Recommend?searchword={quote(keyword)}"
    data = get_json(url)
    if not data:
        raise ValueError("검색 결과 없음")
    return data[0]["eid"]

def fetch_article(eid: str):
    """글 상세 페이지 파싱 -> (title, summary_paras, sections, url)"""
    url = f"{BASE}/Article/{eid}"
    soup = get_html(url)

    # 제목
    title_el = soup.select_one(".meta-head .meta-title") or soup.select_one(".print-title")
    title = title_el.get_text(strip=True) if title_el else eid

    # 요약 (문단)
    smry_box = soup.select_one("#cm_smry .text-detail")
    summary_paras = extract_paragraphs(smry_box)

    # 섹션들 (문단)
    sections = []
    for sec in soup.select("div.detail-section[id^='section-']"):
        head = sec.select_one(".section-head .section-title")
        body = sec.select_one(".section-body")
        title_txt = head.get_text(strip=True) if head else ""
        paras = extract_paragraphs(body)
        if paras:
            sections.append({"title": title_txt, "paragraphs": paras})

    return title, summary_paras, sections, url, soup

def fetch_related_depth1(soup: BeautifulSoup):
    """관련 항목(Depth1) 목록 수집 -> [(eid, title)]"""
    related = []
    # 관련 항목 블록
    rel_ul = soup.select_one("#detail_related_list")
    if not rel_ul:
        return related

    for a in rel_ul.select("a[href^='/Article/']"):
        href = a.get("href", "")
        # /Article/E000000 처럼 끝 eid 추출
        m = re.search(r"/Article/([A-Z0-9]+)", href)
        if not m:
            continue
        eid = m.group(1)
        # 링크 내 표시 제목
        t_el = a.select_one(".tit") or a
        title = clean_text(t_el)
        if not title:
            title = eid
        related.append((eid, title))
    # 중복 제거(같은 eid 우선 1회)
    seen = set()
    uniq = []
    for e, t in related:
        if e not in seen:
            seen.add(e)
            uniq.append((e, t))
    return uniq

# -----------------------------
# 저장
# -----------------------------
def save_article_txt(filepath: str, title: str, url: str, summary_paras: list[str], sections: list[dict]):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{title}\n")
        f.write(f"URL: {url}\n\n")

        if summary_paras:
            f.write("【내용 요약】\n")
            for p in summary_paras:
                f.write(p + "\n\n")

        for sec in sections:
            st = sec.get("title", "").strip()
            if st:
                f.write(f"【{st}】\n")
            for p in sec.get("paragraphs", []):
                f.write(p + "\n\n")

# -----------------------------
# 엔트리 포인트
# -----------------------------
def crawl_keyword(keyword: str, out_dir: str = "outputs", include_related: bool = True, delay_sec: float = 1.2):
    # 1) 메인 글
    eid = first_article_eid(keyword)
    title, smry, sections, url, soup = fetch_article(eid)
    main_name = slugify(title if title else keyword)
    main_path = os.path.join(out_dir, f"{main_name}.txt")
    save_article_txt(main_path, title, url, smry, sections)
    print(f"[저장] {main_path}")

    if not include_related:
        return

    # 2) 관련 항목(depth1)들
    related = fetch_related_depth1(soup)
    if not related:
        print("[정보] 관련 항목 없음")
        return

    # 관련 항목들은 별도 하위 폴더에 저장
    rel_dir = os.path.join(out_dir, f"{main_name}_관련항목")
    os.makedirs(rel_dir, exist_ok=True)

    for i, (reid, rtitle) in enumerate(related, 1):
        try:
            time.sleep(delay_sec)
            r_title, r_smry, r_sections, r_url, _ = fetch_article(reid)
            fname = slugify(r_title)
            path = os.path.join(rel_dir, f"{fname}.txt")
            save_article_txt(path, r_title, r_url, r_smry, r_sections)
            print(f"  └─[저장] ({i}/{len(related)}) {path}")
        except Exception as e:
            print(f"  └─[실패] {rtitle} ({reid}) -> {e}")

if __name__ == "__main__":
    kw = input("검색어를 입력하세요: ").strip()
    if not kw:
        kw = "세종"
    crawl_keyword(kw, out_dir="outputs", include_related=True, delay_sec=1.0)
