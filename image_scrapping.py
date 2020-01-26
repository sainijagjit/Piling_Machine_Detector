#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:33:36 2020

@author: sainijagjit
"""

import pandas as pd
import numpy as np
import time
import requests
import io
from PIL import Image
import os
import hashlib
from selenium import webdriver


DRIVER_PATH = '/home/sainijagjit/Desktop/Darknet/customindz/chromedriver 2/chromedriver'
wd = webdriver.Chrome(executable_path=DRIVER_PATH)




#specifying the data type of input of parameters of our function.
#query = search term like dog, cat
#wb = WebDriver is an automation testing tool. When I say automation, it means it automates test scripts
#sleep_between_interations = sleeping tool

def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        #window.scroll(horizontal scroll, vertical scroll)
        #document.body.scrollHeight - will give height of the entire document
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    
    
    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    #fitting the variables where {}  with the query
    wd.get(search_url.format(q=query))

    # A set is an unordered collection of items. Every element is unique (no duplicates) 
    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        # img : tag , rg_ic : class
        thumbnail_results = wd.find_elements_by_css_selector("img.rg_ic")
        number_results = len(thumbnail_results)
        
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls    
            actual_images = wd.find_elements_by_css_selector('img.irc_mi')
            for actual_image in actual_images:
                if actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(0)
            load_more_button = wd.find_element_by_css_selector(".ksb")
            if load_more_button:
                wd.execute_script("document.querySelector('.ksb').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls


#foler_path : Folder where the image is going to be saved
def persist_image(folder_path:str,url:str,name:int):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        #io will keep the variables in ram
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        #hashlib.sha1(image_content)
        #<sha1 HASH object @ 0x7f149811a6f0>
        
        #hashlib.sha1(image_content).hexdigest()
        #'59772049a05762780221e8a7722f779d84f549ef
        #taking it's first 10 digits
        
        file_path = os.path.join(folder_path,str(name) + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")
        
        

def search_and_download(search_term:str,driver_path:str,target_path='./images',number_images=5):
    #search term : high power cranes
    #output = ./images/high_power_cranes
    target_folder = os.path.join(target_path,'_'.join(search_term.lower().split(' ')))

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with webdriver.Chrome(executable_path=driver_path) as wd:
        res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0)
    i=0 
    for elem in res:
        persist_image(target_folder,elem,i)
        i+=1
 
    




#__name__ will give main if the  file is not imported, will give name of the file if it is imported
if __name__ == "__main__":
    search_term='piling machine'
    search_and_download(search_term=search_term,driver_path=DRIVER_PATH,number_images=700)