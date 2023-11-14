# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    """
    # Don't change this function body!
    # No Python required; create the HTML file.
    return


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def download_page(i):
    url = f'https://books.toscrape.com/catalogue/page-{i}.html'
    request = requests.get(url)
    return request.content

def extract_book_links(text):
    soup = bs4.BeautifulSoup(text, features='html.parser')
    books = soup.find_all('article', class_='product_pod')
    book_links =[]
    for book in books:
        p_rating = book.find('p', class_='star-rating')
        if 'Four' not in p_rating.get('class') and 'Five' not in p_rating.get('class'):
            continue
        price = book.find('p', class_='price_color').text
        price_value = float(price.split('£')[1])
        if price_value >= 50:
            continue
        link = book.find('h3').find('a').get('href')
        book_links.append(link)
    return book_links


def get_product_info(text, categories):
    soup = bs4.BeautifulSoup(text, features='html.parser')
    category = soup.find('ul', class_='breadcrumb').find_all('li')[2].get_text(strip=True)
    if category in categories:
        product_info = {}
        info_tables = soup.find_all('table', class_='table table-striped')
        info_table = info_tables[0] 
        rows = info_table.find_all('tr')
        for row in rows:
                key = row.find('th').get_text(strip=True)
                value = row.find('td').get_text(strip=True)
                product_info[key] = value
                product_info['Category'] = category
                product_info['Rating'] = soup.find('p', class_='star-rating').get('class')[1]
                product_info['Description'] = soup.find('div', id='product_description').find_next_sibling().get_text(strip=True)
                product_info['Title'] = soup.find('div', class_='col-sm-6 product_main').find('h1').get_text(strip=True)
    else:
        return None
    return product_info

def scrape_books(k, categories):
    results = pd.DataFrame()
    book_links = []
    for i in range(1, k+1):
        page_soup = download_page(str(i))
        book_links = extract_book_links(page_soup)
        for link in book_links:
            book_url = f'https://books.toscrape.com/catalogue/{link}'
            request = requests.get(book_url).content
            prodInfo = get_product_info(request, categories)
            if prodInfo is not None:
                results = results.append(prodInfo, ignore_index=True)
    return results


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------
                                  

def stock_history(ticker, year, month):
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month+1:02d}-01" if month != 12 else f"{year+1}-01-01"
    date_range = pd.date_range(start=start_date, end=end_date, closed='left')
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&to={end_date}&apikey=Tiq0MdfbMMWj2qVheY8kM0IMz8LDBcDM"
    r = requests.get(url)
    rr = pd.DataFrame(r.json())
    historical_data = rr['historical'] if 'historical' in rr else []
    df = pd.json_normalize(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].isin(date_range)]
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df


def stock_stats(history):
    start_price = history.iloc[-1]['open']
    end_price = history.iloc[0]['close']
    percent_change = ((end_price - start_price) / start_price) * 100
    percent_change_str = f"{percent_change:+.2f}%"

    daily_volume_billion = (history['high'] + history['low']) / 2 * history['volume'] / 1e9
    total_volume_billion = daily_volume_billion.sum()
    total_volume_billion_str = f"{total_volume_billion:.2f}B"
    return (percent_change_str, total_volume_billion_str)

# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def formal_url(storyid):
    return f'https://hacker-news.firebaseio.com/v0/item/{storyid}.json'


def process_comment(commentid, comment_datas):
    comment_data = requests.get(formal_url(commentid)).json()
    if comment_data.get("dead"):
        return
    comment_datas.append({
        'id': comment_data['id'],
        'by': comment_data.get('by', ''),
        'text': comment_data.get('text', ''),
        'parent': comment_data.get('parent', ''),
        'time': pd.Timestamp(comment_data['time'], unit='s')
    })
    if 'kids' in comment_data:
        for child_commentid in comment_data['kids']:
            process_comment(child_commentid, comment_datas)


def get_comments(storyid):
    r = requests.get(formal_url(storyid)).json()
    comment_datas = []
    for commentid in r.get('kids', []):
        process_comment(commentid, comment_datas)
    comments_df = pd.DataFrame(comment_datas)
    return comments_df