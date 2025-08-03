import requests
from bs4 import BeautifulSoup
import trafilatura
import pytesseract
from PIL import Image
import json
import os

BASE_URL = "https://arxiv.org"
CATEGORY = "cs.CL"
LIMIT = 200

def fetch_abs_urls(category, limit=200):
    base = f"https://arxiv.org/list/{category}/new"
    r = requests.get(base)
    soup = BeautifulSoup(r.text, 'html.parser')
    links = soup.select('dt a[href^="/abs/"]')
    urls = list({BASE_URL + link['href'] for link in links})
    return urls[:limit]

def scrape_and_clean(url):
    r = requests.get(url)
    downloaded = trafilatura.extract(r.text, output_format='json', include_comments=False)
    return downloaded

def ocr_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

def save_json(data, filename="arxiv_clean.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    urls = fetch_abs_urls(CATEGORY, LIMIT)
    output = []
    for url in urls:
        try:
            cleaned = scrape_and_clean(url)
            if not cleaned:
                continue
            parsed = json.loads(cleaned)
            output.append({
                "url": url,
                "title": parsed.get("title", ""),
                "abstract": parsed.get("text", ""),
                "authors": parsed.get("author", ""),
                "date": parsed.get("date", "")
            })
        except Exception as e:
            print(f"[!] Failed {url}: {e}")
    save_json(output)

if __name__ == "__main__":
    main()
