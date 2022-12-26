"""
Extract readable text from a URL via various hacks
"""

from loguru import logger
from bs4 import BeautifulSoup, NavigableString, Tag
from readability import Document    # https://github.com/buriy/python-readability
import requests
import urllib.parse

from github_api import github_readme_text
from pdf_text import pdf_text

from exceptions import *
    

def extract_text_from_html(content):
    soup = BeautifulSoup(content, 'html.parser')

    output =  ""
    title = soup.find('title')
    if title:
        output += "Title: " + title
        
    blacklist = ['[document]','noscript','header','html','meta','head','input','script', "style"]
    # there may be more elements we don't want
    
    for t in soup.find_all(text=True):
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    return output


def get_url_text(url):
    """
    get url content and extract readable text
    returns the text
    """
    resp = requests.get(url, timeout=30)

    if resp.status_code != 200:
        logger.warning(url)
        raise NetworkError(f"Unable to get URL ({resp.status_code})")

    CONTENT_TYPE = resp.headers['Content-Type']
    
    if 'pdf' in CONTENT_TYPE:
        return pdf_text(resp.content)
    
    if "html" not in CONTENT_TYPE:
        logger.warning(url)
        raise UnsupportedContentType(f"Unsupported content type: {resp.headers['Content-Type']}")

    doc = Document(resp.text)
    text = extract_text_from_html(doc.summary())

    if not len(text) or text.isspace():
        logger.warning(url)
        raise EmptyText("Unable to extract text data from url")
    return text



def url_to_text(url):
    HOPELESS = ["youtube.com",
                "www.youtube.com"]
    if urllib.parse.urlparse(url).netloc in HOPELESS:
        logger.warning(url)        
        raise UnsupportedHostException("Unsupported host: {urllib.parse.urlparse(url).netloc}")

    if urllib.parse.urlparse(url).netloc == 'github.com':
        # for github repos use api to attempt to find a readme file
        text = github_readme_text(url)
    else:
        text = get_url_text(url)

    logger.info("url_to_text: "+text)
    return text
