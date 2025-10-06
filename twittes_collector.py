# selenium_scraper_improved.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd


def extract_text_from_article(article):
    # tweet text lives in one or more divs with a lang attribute
    parts = article.find_elements(By.XPATH, ".//div[@lang]")
    texts = []
    for p in parts:
        try:
            t = p.text
            if t:
                texts.append(t.strip())
        except Exception:
            continue
    return "\n".join(texts).strip()


def extract_tweet_link(article):
    # find anchor with /status/ in href
    try:
        a = article.find_element(By.XPATH, ".//a[contains(@href, '/status/')]")
        href = a.get_attribute("href")
        return href
    except Exception:
        return None


def extract_datetime(article):
    try:
        t = article.find_element(By.TAG_NAME, "time")
        return t.get_attribute("datetime")
    except Exception:
        return None


def scrape_user_page(username, limit=50, max_scrolls=30, headless=True):
    url = f"https://twitter.com/{username}"
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.binary_location = "/usr/bin/chromium"

    driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"), options=options)
    wait = WebDriverWait(driver, 15)

    driver.get(url)

    # Wait for profile header or first article to appear
    try:
        wait.until(EC.any_of(
            EC.presence_of_element_located((By.TAG_NAME, "article")),
            EC.presence_of_element_located((By.XPATH, "//div[contains(@data-testid,'primaryColumn')]"))
        ))
    except Exception:
        # page might require login or be blocked
        print("Warning: profile didn't load normally. You might need to log in or Twitter/X changed the layout.")

    tweets = []
    seen_ids = set()
    scrolls = 0

    while len(tweets) < limit and scrolls < max_scrolls:
        # find all article nodes currently loaded
        articles = driver.find_elements(By.XPATH, "//article")
        for art in articles:
            try:
                tweet_link = extract_tweet_link(art)
                if not tweet_link:
                    continue
                # extract tweet id from link (last part after /status/)
                tweet_id = tweet_link.rstrip("/").split("/")[-1]
                if tweet_id in seen_ids:
                    continue

                text = extract_text_from_article(art)
                created_at = extract_datetime(art)
                tweets.append({
                    "id": tweet_id,
                    "link": tweet_link,
                    "datetime": created_at,
                    "text": text
                })
                seen_ids.add(tweet_id)
                if len(tweets) >= limit:
                    break
            except Exception:
                continue

        if len(tweets) >= limit:
            break

        # scroll down to load more tweets
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.2)  # give JS time to fetch more tweets
        scrolls += 1

    driver.quit()
    return pd.DataFrame(tweets)


if __name__ == "__main__":
    df = scrape_user_page("elonmusk", limit=20, max_scrolls=25)
    df.to_csv("selenium_tweets_improved.csv", index=False)
    print("Scraped", len(df), "tweets")
