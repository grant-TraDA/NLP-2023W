import requests
import json
import time
from datetime import date, timedelta

# TODO: provide credentials
USERNAME = ""
PASSWORD = ""

URL_ONE_NEWS = ""
URL_ALL_NEWS_DATE = ""

START_DATE = date(2023, 10, 6)
END_DATE = date(2023, 10, 31)

def download_one_news(news_id: int) -> dict:
    url = URL_ONE_NEWS + str(news_id)
    response = requests.get(url, auth=(USERNAME, PASSWORD))
    response_dict = json.loads(response.text)
    return response_dict


def list_news_from_date(date: str, language: str = "en") -> list[int]:
    # date = "10.11.2022"
    url = URL_ALL_NEWS_DATE + language + '/' + date
    response = requests.get(url, auth=(USERNAME, PASSWORD))
    response_list = json.loads(response.text)
    return response_list


def download_news_from_date(date: str, language: str = "en", omit_categories: list = None) -> list:
    news_ids = list_news_from_date(date=date, language=language)
    downloaded_news = []
    for news_id in news_ids:
        news = download_one_news(news_id)
        omit = False
        if omit_categories is not None:
            for category in omit_categories:
                if category in news["categories"]:
                    omit = True
                    break
        if not omit:
            downloaded_news.append(news)
    return downloaded_news


def date_range(start_date: str, end_date: str):
    start_date = date(2019, 1, 1)
    end_date = date(2020, 1, 1)
    delta = timedelta(days=1)
    while start_date <= end_date:
        print(start_date.strftime("%Y-%m-%d"))
        start_date += delta


def download_news_from_dates_range(start_date: date, end_date: date, language: str = "en",
                                   omit_categories: list = None, save: bool = True,
                                   sleep_time: float = 0) -> list:
    delta = timedelta(days=1)
    filename = f"news_from_{start_date.strftime('%Y-%m-%d')}_to_"
    news_to_save = []
    while start_date <= end_date:
        time.sleep(sleep_time)
        news_date = start_date.strftime("%d.%m.%Y")
        try:
            downloaded_news = download_news_from_date(date=news_date, language=language,
                                                      omit_categories=omit_categories)
        except Exception as e:
            print(e)
            print(f"Due to an error saved news only to {(start_date - delta).strftime('%Y-%m-%d')}")
            break
        news_to_save += downloaded_news
        print(f"Downloaded data from {start_date.strftime('%Y-%m-%d')}")
        start_date += delta
    filename += f"{(start_date - delta).strftime('%Y-%m-%d')}.json"
    if save:
        with open("raw_data/" + language + '/' + filename, "w") as file:
            json.dump(news_to_save, file, indent=2)
    return news_to_save


if __name__ == '__main__':
    # example
    s_date = START_DATE
    e_date = END_DATE
    _ = download_news_from_dates_range(s_date, e_date, language="sl",
                                       omit_categories=None, sleep_time=3)
