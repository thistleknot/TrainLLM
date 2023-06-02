#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import pandas as pd
quotes = []

parent_url = "https://graciousquotes.com/all/"
response = requests.get(parent_url)

if response.status_code == 200:
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find all h3 elements with class 'wp-block-heading'
    h3_elements = soup.find_all('h3', class_='wp-block-heading')
    
    for h3 in h3_elements:
        section_title = h3.get_text(strip=True)
        print(f"Processing section: {section_title}")
        
        # Find all the links under the current section
        p_element = h3.find_next_sibling('p')
        links = p_element.find_all('a')
        
        for link in links:
            child_url = link['href']
            print(f"Processing link: {child_url}")
            
            child_response = requests.get(child_url)
            if child_response.status_code == 200:
                child_html = child_response.text
                child_soup = BeautifulSoup(child_html, 'html.parser')
                
                # Extract quotes from child page
                figcaptions = child_soup.find_all('figcaption')
                for figcaption in figcaptions:
                    quotes.append(figcaption.text.strip())
                    print(figcaption.text.strip())
            else:
                print(f"Failed to retrieve the child page: {child_url}")
else:
    print("Failed to retrieve the parent page.")
    
pd.DataFrame(quotes).to_csv('graciousquotes.csv')


# In[ ]:





# In[ ]:


import requests
import pandas as pd
from bs4 import BeautifulSoup

# Data
BASE_URL = "https://www.azquotes.com"
AUTHORS_URL = f"{BASE_URL}/quotes/authors.html"
AUTHOR_SINGLE_URL = f"{BASE_URL}/quotes/authors"
QUOTES_BY_AUTHOR_URL = f"{BASE_URL}/author"

# Selectors
AUTHORS_TABLES_SELECTOR = ".authors-page ul"
AUTHORS_LIST_SELECTOR = ".leftcol-inner .table tbody tr"
AUTHORS_PAGINATION_SELECTOR = ".table + .pager li"
AUTHOR_SINGLE_PAGINATION_SELECTOR = ".pager li"
QUOTES_LIST_SELECTOR = "ul.list-quotes li"

# Get all authors
def get_authors():
    response = requests.get(AUTHORS_URL)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        authors_tables = soup.select(AUTHORS_TABLES_SELECTOR)
        authors = []
        for table in authors_tables:
            links = table.find_all('a')
            for link in links:
                authors.append({'name': link.text.strip(), 'link': link['href'].replace('/author/', '')})
        return authors
    return []

# Get the number of pages for an author
def get_author_pagination(author_link):
    response = requests.get(f"{AUTHOR_SINGLE_URL}/{author_link}")
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        pagination = soup.select(AUTHOR_SINGLE_PAGINATION_SELECTOR)
        pages = []
        for page in pagination:
            page_text = page.find('a').text
            if page_text.isdigit():
                pages.append(int(page_text))
        return max(pages) if pages else 1
    return 1

# Get quotes by author
# Get quotes by author
def get_quotes_by_author(author_link, page=1):
    response = requests.get(f"{QUOTES_BY_AUTHOR_URL}/{author_link}/{page}")
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        quote_lis = soup.select(QUOTES_LIST_SELECTOR)
        quotes = []
        for quote_li in quote_lis:
            if not quote_li.find("div", class_="wrap-block-for-ad"):
                quote_a = quote_li.find('a')
                if quote_a:
                    quotes.append(quote_a.text.strip())
                else:
                    print(f"Warning: No 'a' tag found in quote_li: {quote_li}")
        return quotes
    return []

# Main script
authors = get_authors()
quotes = []

for author in authors:
    author_link = author['link']
    num_pages = get_author_pagination(author_link)

    for page in range(1, num_pages + 1):
        author_quotes = get_quotes_by_author(author_link, page)
        quotes.extend(author_quotes)

# Save quotes to a CSV file
pd.DataFrame(quotes).to_csv('quotes_by_author.csv')


# In[ ]:




