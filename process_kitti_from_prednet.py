import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import numpy as np

DATA_DIR = './kitti_data/'

categories = ['city', 'residential', 'road']

# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
val_recordings = [('city', '2011_09_26_drive_0005_sync')]
test_recordings = [('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)

# Download raw zip files by scraping KITTI website
def download_data():
    base_dir = os.path.join(DATA_DIR, 'raw/')
    if not os.path.exists(base_dir): os.mkdir(base_dir)
    for c in categories:
        url = "http://www.cvlibs.net/datasets/kitti/raw_data.php?type=" + c
        r = requests.get(url)
        soup = BeautifulSoup(r.content,"lxml")
        drive_list = soup.find_all("h3")
        drive_list = [d.text[:d.text.find(' ')] for d in drive_list]
        print( "Downloading set: " + c)
        c_dir = base_dir + c + '/'
        if not os.path.exists(c_dir): os.mkdir(c_dir)
        for i, d in enumerate(drive_list):
            print( str(i+1) + '/' + str(len(drive_list)) + ": " + d)
            url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/" + d + "/" + d + "_sync.zip"
            urllib.request.urlretrieve(url, filename=c_dir + d + "_sync.zip")

if __name__ == '__main__':
    download_data()