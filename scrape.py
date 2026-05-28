import requests
import json
import time
import re
from bs4 import BeautifulSoup


def scrape_samsung_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "html.parser")

        # Title
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title else "No title"

        for tag in soup.find_all(["nav", "footer", "header"]):
            tag.decompose()
        for tag in soup.find_all(class_=re.compile(
                r"footer|header|nav|menu|breadcrumb|cookie|banner|feedback|recommend", re.I)):
            tag.decompose()

        # Collect h2, h3, p, li
        content_parts = []
        seen = set()

        for tag in soup.find_all(["h2", "h3", "p", "li"]):
            text = tag.get_text(separator=" ", strip=True)

            if (len(text) > 20
                    and text not in seen
                    and "javascript" not in text.lower()):

                # Add Subtitle (with ##)
                if tag.name in ["h2", "h3"]:
                    content_parts.append(f"\n## {text}")
                else:
                    content_parts.append(text)
                seen.add(text)

        content = "\n".join(content_parts)

        return {
            "url": url,
            "title": title_text,
            "content": content
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


example_urls = [
    "https://www.samsung.com/us/support/answer/ANS10001572/",
    "https://www.samsung.com/us/support/answer/ANS10002537/",
    "https://www.samsung.com/us/support/answer/ANS10003231/",
    "https://www.samsung.com/us/support/answer/ANS10003416/"
    "https://www.samsung.com/us/support/answer/ANS10003662/",
    "https://www.samsung.com/us/support/troubleshoot/TSG10001548/",
    "https://www.samsung.com/us/support/troubleshoot/TSG10001565/",
    "https://www.samsung.com/us/support/troubleshoot/TSG10001468/",
    "https://www.samsung.com/us/support/troubleshoot/TSG10001451/",
    "https://www.samsung.com/us/support/troubleshoot/TSG10001446/",
    "https://www.samsung.com/us/support/troubleshoot/TSG10001529/",
    "https://www.samsung.com/us/support/troubleshoot/TSG10002028/",
    "https://www.samsung.com/us/support/troubleshoot/TSG10002488/",
    "https://www.samsung.com/us/support/troubleshoot/TSG10004671/",
    "https://www.samsung.com/us/support/troubleshoot/TSG10004073/",
    "https://www.samsung.com/us/support/troubleshoot/TSG10004905/",
]

count = 0
for url in example_urls:
    print(f"Scraping: {url}")
    data = scrape_samsung_page(url)
    if data:
        count += 1
        print(f"✓ {data['title']}")
        with open("samsung_docs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    time.sleep(1)

print(f"\nTotal {count} Urls scraped → samsung_docs.jsonl")
