import requests
from requests.auth import HTTPBasicAuth
import json
from datetime import datetime, timedelta

# Function to fetch article IDs for a specific language and date
def get_article_ids(language, date):     # language : sl / fa / en
    url = f'https://api.sta.si/news/{language}/{date}'
    response = requests.get(url, auth=HTTPBasicAuth('', ''))
    if response.status_code == 200:
        articles_ids = response.json()
        return articles_ids
    else:
        print(f"Failed to fetch {language} articles for {date}. Status code: {response.status_code}")
        return []

# Function to get article content by ID
def get_article_content(language, article_id):   # language: sta / fa
    url = f'https://api.sta.si/news/{language}/{article_id}'
    response = requests.get(url, auth=HTTPBasicAuth('', ''))
    if response.status_code == 200:
        article_content = response.json()
        return article_content
    else:
        print(f"Failed to fetch {language} content for article ID {article_id}. Status code: {response.status_code}")
        return {}

# Function to save content to a JSON file
def save_to_json(file_name, content):
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(content, json_file, ensure_ascii=False, indent=4)



if __name__ == '__main__':

    # Get today's date and calculate the date several days ago
    today = datetime.now().date()
    starting_date = today - timedelta(days=1)

    # Fetch article IDs for Slovenian articles
    slovenian_article_ids = []
    current_date = starting_date
    while current_date <= today:
        date_str = current_date.strftime('%d.%m.%Y')
        slovenian_article_ids.extend(get_article_ids('sl', date_str))
        current_date += timedelta(days=1)

    # Fetch article IDs for English articles
    english_article_ids = []
    current_date = starting_date
    while current_date <= today:
        date_str = current_date.strftime('%d.%m.%Y')
        english_article_ids.extend(get_article_ids('fa', date_str))
        current_date += timedelta(days=1)

    # Fetch slovenian content for STA articles using obtained IDs
    sta_articles_content = {}
    for article_id in slovenian_article_ids:
        article_content = get_article_content('sta', article_id)
        sta_articles_content[article_id] = article_content

    # Save STA slovenian article content to a JSON file
    save_to_json('sta_articles_content.json', sta_articles_content)

    # Fetch english content for STA articles using obtained IDs
    fa_articles_content = {}
    for article_id in english_article_ids:
        article_content = get_article_content('fa', article_id)
        fa_articles_content[article_id] = article_content

    # Save english STA article content to a JSON file
    save_to_json('fa_articles_content.json', fa_articles_content)