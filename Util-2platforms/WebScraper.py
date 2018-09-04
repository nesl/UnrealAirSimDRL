from urllib.request import Request, urlopen, urlretrieve
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup
from PIL import Image
import os
import requests

def make_soup(url):
    website = urlopen(url) # returns HTML data
    soupdata = BeautifulSoup(website, "html.parser")
    return soupdata # Returns the soup HTML object

def download_image_junk(url, filepath, img_name):
    url = str(url)
    img_name = str(img_name)
    soup = make_soup(str(url))
    i = 0
    for img in soup.findAll("img"):
        print("Image found in url")
        image = img.get('src')
        if image[:1] == "/":
            image = url + image
        else:
            # Save the image
            filename = filepath + "\\" + str(img_name) + "_" + str(i) + ".jpeg"
            imagefile = open(filename, "bw")
            imagefile.write(urlopen(image).read())
            imagefile.close()
            print("Image ", str(img_name), "_", i, "saved in directory")
            i = i + 1

def download_images(urls, filepath, img_names, session = None):
    if not session:
        session = requests.Session()
    for link, img_name in zip(urls, img_names):
        try:
            r = session.get(link)
        except (requests.exceptions.RequestException, UnicodeError) as e:
            print(e)
        print('image', img_name)
        with open( filepath + "\\" + img_name + ".jpeg", "wb") as f:
            f.write(r.content)


def scrape_website(url):
    uClient = urlopen(url) # Connect to website
    page_html = uClient.read() # read html file
    uClient.close() # close website connection
    # Parse the page_html with soup
    soup = BeautifulSoup(page_html, "html.parser")
    print(soup.h1) # display the website's html information
    # Traverse the soup
    return soup
    
    

#url = 'https://www.flickr.com/photos/tags'
#scrape_website(url)
#url = "https://www.newbiehack.com"
#soup = scrape_website(url)








