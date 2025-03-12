# -*- coding: utf-8 -*-
"""progres1_HLS.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lPC5yRSc3j-Vl-Q8GLyYz_5y509jUloC
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

base_url = "https://sinta.kemdikbud.go.id/affiliations/profile/398/?page="
view_param = "&view=scopus"
start_page = 1
end_page = 3
data = []

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

with requests.Session() as session:
    session.headers.update(headers)

    for page in range(start_page, end_page + 1):
        url = base_url + str(page) + view_param
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.find_all("div", class_="ar-list-item mb-5")

            for item in articles:
                title_tag = item.find("div", class_="ar-title").find("a")
                title = title_tag.text.strip() if title_tag else "Tidak ditemukan"
                link = title_tag["href"] if title_tag else "Tidak ditemukan"

                creator_tag = item.find("a", string=re.compile(r"Creator\s*:", re.I))
                creator = "Tidak ditemukan"
                if creator_tag:
                    creator = creator_tag.text.replace("Creator :", "").strip()

                journal_tag = item.find("a", class_="ar-pub")
                journal_name = journal_tag.text.strip() if journal_tag else "Tidak ditemukan"

                year_tag = item.find("a", class_="ar-year")
                year = year_tag.text.strip() if year_tag else "Tidak ditemukan"

                cited_tag = item.find("a", class_="ar-cited")
                citations = "0"
                if cited_tag:
                    cited_text = cited_tag.text.strip()
                    match = re.search(r"(\d+)", cited_text)
                    if match:
                        citations = match.group(1)

                data.append([title, link, creator, journal_name, year, citations])

            print(f"craped page {page}")

        except requests.exceptions.RequestException as e:
            print(f"Error pada halaman {page}: {e}")
            time.sleep(random.randint(10, 30))
            continue

        time.sleep(random.randint(5, 15))

# Simpan ke CSV
df = pd.DataFrame(data, columns=["Title", "URL", "Creator", "Journal", "Year", "Citations"])
df.to_csv("sinta_scraped_data.csv", index=False)

print("Scraping selesai, data disimpan dalam sinta_scraped_data.csv!")

df.head()