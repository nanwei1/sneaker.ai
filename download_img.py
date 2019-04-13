import argparse
import json
import itertools
import logging
import re
import os
import uuid
import sys
from urllib2 import urlopen, Request
from bs4 import BeautifulSoup

def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('[%(asctime)s %(levelname)s %(module)s]: %(message)s'))
    logger.addHandler(handler)
    return logger
logger = configure_logging()

REQUEST_HEADER = {
    'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

def get_soup(url, header):
    response = urlopen(Request(url, headers=header))
    return BeautifulSoup(response, 'html.parser')

def get_query_url(query):
    return "https://www.google.co.in/search?q=%s&source=lnms&tbm=isch" % query

def extract_images_from_soup(soup):
    image_elements = soup.find_all("div", {"class": "rg_meta"})
    metadata_dicts = (json.loads(e.text) for e in image_elements)
    link_type_records = ((d["ou"], d["ity"]) for d in metadata_dicts)
    return link_type_records

def extract_images(query, num_images, verbose):
    url = get_query_url(query)
    if verbose: logger.info("Souping")
    soup = get_soup(url, REQUEST_HEADER)
    if verbose: logger.info("Extracting image urls")
    link_type_records = extract_images_from_soup(soup)
    return itertools.islice(link_type_records, num_images)

def get_raw_image(url):
    req = Request(url, headers=REQUEST_HEADER)
    resp = urlopen(req)
    return resp.read()

def save_image(raw_image, image_type, save_directory):
    extension = image_type if image_type else 'jpg'
    file_name = str(uuid.uuid4().hex) + "." + extension
    save_path = os.path.join(save_directory, file_name)
    with open(save_path, 'wb+') as image_file:
        image_file.write(raw_image)

def download_images_to_dir(images, save_directory, num_images, verbose):
    if not os.path.exists(save_directory):
    	os.makedirs(save_directory)
    print("Directory " , save_directory ,  " Created ")
    for i, (url, image_type) in enumerate(images):
        try:
            if verbose: logger.info("Making request (%d/%d): %s", i, num_images, url)
            raw_image = get_raw_image(url)
            save_image(raw_image, image_type, save_directory)
        except Exception as e:
            if verbose: logger.exception(e)
            
def run_downloads(query, save_directory, num_images=1, verbose=True):
    query = '+'.join(query.split())
    if verbose: logger.info("Extracting image links")
    images = extract_images(query, num_images, verbose)
    if verbose: logger.info("Downloading images")
    download_images_to_dir(images, save_directory, num_images, verbose)
    if verbose: logger.info("Finished")

def start_download(search = "cat", num_images = 1, verbose=True):
	directory = os.path.join(os.getcwd(), "downloaded_data/"+search);
	run_downloads(search, directory, num_images, verbose)
	
	
