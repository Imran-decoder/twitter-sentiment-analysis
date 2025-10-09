import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Configure Chrome options
def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    # Path for Streamlit Cloud (chromium + chromedriver)
    chrome_bin = "/usr/bin/chromium-browser"
    driver_bin = "/usr/bin/chromedriver"

    if os.path.exists(chrome_bin) and os.path.exists(driver_bin):
        print("✅ Using system Chromium and Chromedriver")
        options.binary_location = chrome_bin
        service = Service(driver_bin)
        driver = webdriver.Chrome(service=service, options=options)
    else:
        print("⚙️ Using webdriver_manager fallback")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    return driver

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd


def scrape_user_page(username, limit=50, max_scrolls=30, headless=True):
    url = f"https://twitter.com/{username}"
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")

    # ✅ webdriver_manager handles ChromeDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 15)

    driver.get(url)

    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "article")))
    except:
        print("Warning: profile didn't load. Twitter/X may require login.")

    tweets, seen_ids, scrolls = [], set(), 0

    while len(tweets) < limit and scrolls < max_scrolls:
        articles = driver.find_elements(By.XPATH, "//article")
        for art in articles:
            try:
                link_elem = art.find_element(By.XPATH, ".//a[contains(@href, '/status/')]")
                tweet_link = link_elem.get_attribute("href")
                tweet_id = tweet_link.rstrip("/").split("/")[-1]

                if tweet_id in seen_ids:
                    continue

                text_parts = art.find_elements(By.XPATH, ".//div[@lang]")
                text = "\n".join([p.text for p in text_parts if p.text.strip()])

                try:
                    created_at = art.find_element(By.TAG_NAME, "time").get_attribute("datetime")
                except:
                    created_at = None

                tweets.append({
                    "id": tweet_id,
                    "link": tweet_link,
                    "datetime": created_at,
                    "text": text
                })
                seen_ids.add(tweet_id)

                if len(tweets) >= limit:
                    break
            except:
                continue

        if len(tweets) >= limit:
            break

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
        scrolls += 1

    driver.quit()
    return pd.DataFrame(tweets)
