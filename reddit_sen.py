import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def fetch_reddit_posts(subreddit="technology", limit=20, scroll_pause=2):
    # --- Chrome setup ---
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(f"https://www.reddit.com/r/{subreddit}/")

    print(f"Fetching posts from r/{subreddit}...")
    # time.sleep(5)

    data, seen = [], set()
    scrolls = 0

    while len(data) < limit and scrolls < 100:
        posts = driver.find_elements(By.CSS_SELECTOR, "div[data-testid='post-container']")
        for post in posts:
            try:
                post_id = post.get_attribute("id") or f"reddit_{len(data)}"
                if post_id in seen:
                    continue

                # --- Title ---
                try:
                    title = post.find_element(By.CSS_SELECTOR, "h3").text
                except:
                    title = None

                # --- Upvotes ---
                try:
                    score = post.find_element(By.CSS_SELECTOR, "div[data-click-id='upvote']").text
                except:
                    score = None

                # --- Post URL ---
                try:
                    link = post.find_element(By.CSS_SELECTOR, "a[data-testid='post-title']")
                    post_url = link.get_attribute("href")
                except:
                    post_url = None

                if title:
                    data.append({
                        "id": post_id,
                        "title": title,
                        "score": score,
                        "url": post_url
                    })
                    seen.add(post_id)

                if len(data) >= limit:
                    break

            except Exception as e:
                continue

        # Scroll to load more
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)
        scrolls += 1

    driver.quit()

    if not data:
        print("⚠️ No posts found. Try increasing scrolls or removing headless mode.")
    else:
        print(f"✅ Scraped {len(data)} posts.")

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = fetch_reddit_posts("technology", limit=10)
    print(df)
    df.to_csv("reddit_posts.csv", index=False)
