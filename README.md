This is a toy personal project in which I am trying to develop an algorithm that can tell user the what sneaker it is given an image. Work in progress.

## Scrap Images:
The full-sized images are scraped from Google images with help from [this gist](https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57).

You need python 2.7. Personally I created a conda environment:
```
conda create -n scrape_img python=2.7
```

Activate the new environment and then install BeautifulSoup for web scraping:
```
conda activate scrape_img
conda install -c anaconda beautiful-soup
```

I modified the file from [this gist](https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57) into download_img_backup.py. It can be run like the following:
```
python download_img_backup --search "air_jordan_1" -num_images=100
```

The maximum for num_images here is 100 because Google image by default only shows the first 100 results. This may or may not be enough - I will try transfer learning as well as some one-shot learning techniques such as siamese network first. If the results are not satisfactory, I will need to modify the scraping code to allow for more training data.

I converted the main() of download_img_backup into a function and renamed the file as download_img.py for convenience. Then download_aj.py calls the function 23 times from air_jordan_1 to air_jordan_23 to download 100 images of each.
```
python download_aj
```

The images will be downloaded to "/dowanloaded_data/", with subfolders like "air_jordan_1", "air_jordan_2", etc.

**update 2019/04/13:**
The above returns max of 100 images because that is the maximum number given by Google images without scrolling down or clicking one "Show more results." In case if images are not enough, I started using another code in download_img2.py which allows for downloading more pictures. To use the script, one need selenium package in python and chromedriver. I was using Ubuntu 16.04 with Chrome already installed. 
```
conda install selenium
wget https://chromedriver.storage.googleapis.com/2.41/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
sudo mv chromedriver /usr/bin/chromedriver
sudo chown root:root /usr/bin/chromedriver
sudo chmod +x /usr/bin/chromedriver
```

The script download_img2.py was adapted from the comments in [this gist](https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57) but the original code did not work for me. Had to tweak several places to make it work correctly. One thing to mention here is that sometimes the program get stuck at downloading one image forever, so I added a timeout if one download takes longer than 5 seconds.

Run download_aj the same as before (except now it calls start_download() in the new script download_img2.py). It attempts to download 400 images for each class (sometimes there are fewer results available to download on Google image) which may take a while, so sit back and grab a drink - maybe even watch a movie!


## Pre-processing of data
- Delete corrupt files
- Delete random images (posters, drawings, fish pic?)
### size?
