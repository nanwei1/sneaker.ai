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

## Pre-processing of data
### manually check the images to delete wrong ones
### size?
