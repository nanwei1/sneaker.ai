from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib2
import argparse
import sys
import time
import signal

def handler(signum, frame):
    print "Download taking too long!"
    raise Exception("Download Timeout")

def start_download(search="apples", num_images=200, verbose = False):
    url = "https://www.google.co.in/search?q="+search+"&source=lnms&tbm=isch"
    browser = webdriver.Chrome("/usr/bin/chromedriver")
    browser.get(url)
    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    counter = 0
    succounter = 0
    dir = os.path.join("downloaded_data/", search)

    if not os.path.exists(dir):
        os.mkdir(dir)
    number_of_scrolls = num_images / 400 + 1

    for _ in range(number_of_scrolls):
        for __ in xrange(10):
			# multiple scrolls needed to show all 400 images
			browser.execute_script("window.scrollBy(0, 10000)")
			time.sleep(0.2)

        print len(browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'))
        time.sleep(0.5)
        try:
			browser.find_element_by_xpath("//input[@value='Show more results']").click()
        except Exception as e:
			print "Less images found:", e
			break

    signal.signal(signal.SIGALRM, handler)
    for x in browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
        if succounter>=num_images:
            break
        counter = counter + 1
        if verbose: print "URL:",json.loads(x.get_attribute('innerHTML'))["ou"]

        img = json.loads(x.get_attribute('innerHTML'))["ou"]
        imgtype = json.loads(x.get_attribute('innerHTML'))["ity"]
        # give timeout warning if it takes more than 5sec downloading one image
        signal.alarm(5)
        try:
            req = urllib2.Request(img, headers={'User-Agent': header})
            raw_img = urllib2.urlopen(req).read()
            File = open(os.path.join(dir, search + "_" + str(succounter) + "." + imgtype), "wb")
            File.write(raw_img)
            File.close()
            succounter = succounter + 1
        except:
                print "can't get img"
        if verbose: print "Total Count:", counter
        if verbose: print "Succsessful Count:", succounter
        print "Working on ", search, ", Progress: ", succounter, "/", num_images, "\r"
        # sys.stdout.write("\r{0}>".format("="*int(counter/num_images*10)))
    print succounter, "pictures succesfully downloaded"
    # turn off timer
    signal.alarm(0)
    browser.close()
    time.sleep(0.5)
