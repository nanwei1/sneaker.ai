This is a toy personal project in which I am trying to develop an algorithm that can tell user the what sneaker it is given an image. Work in progress.

## Scrap Images: (2019/04/13)
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





## Prototyping (2019/04/13)
**update 2019/04/14:**
I started off by using only 5 classes (Air Jordan 1 - 5) instead of doing the full 23 classes to get a quick start. If it turns out that I need to download more data or something else at least I can know early

### Pre-processing of data
- Delete corrupt files
- Delete random images (posters, drawings, fish pic?)
- Generate pandas dataframe recording filename and labels - this step is achieved by running "data_prep.ipynb." Here I randomly split the data set into training and test (80% and 20%) respectively. It is better to use a validation set to do hyperparameter tuning, model selection etc. while holding out the test dataset until the very end, but since this is a toy project and I really do not have too much data to start with so this was ignored.

### Model training - CNN from Scratch
[Ipython Notebook](.//cnn.ipynb)
First thing is to make sure the ingested data are of the same size. The images downloaded have a wide variety of types (grayscale, rgb, etc.) and sizes. PyTorch provides tools for this in the "transforms" package. An example would be to use the following transformation
```
transforms.Compose([transforms.ToPILImage(),  # allowing for grayscale, resize, etc but need ToSensor() in the end
                    transforms.Grayscale(),   # convert all images into grayscale 
                    transforms.Resize(size=224),  # resize to 224 by 224
                    transforms.ToTensor()])
```

I initially attempted to train a CNN from scratch with images transformed to grayscale with 224x224 size. Tried a few different setups (architecture, learning rate, batch size, dropout, transformation, etc). A quick recap of the process (see cnn.ipynb for code):

- Started with a bigger net with 5 conv layers with 3 fc layers. Training loss was reducing while test loss when up. It is clear that there is overfitting. [Result here.](.//lr=1e-4_batch=16.png)
- Added dropout at the fc layers to reduce overfitting. Result was barely better. I think the model is too big for dataset this small. [Result here.](.//lr=1e-4_batch=16_dropout.png)
- Recuced to conv layers to 3. Overfitting was greatly reduced, but training was slow. [Result here.](.//smaller_model_lr=1e-4_batch=16_dropout.png)
- Increased learning rate from 1e-4 to 5e-4. Traning is faster now but overfitting appears again. In fact in the previous attemp there is also overfitting but it started toward the end of training so I did not notice. [Result here.](.//smaller_model_lr=5e-4_batch=16_dropout.png)
- How can I further reduce overfitting? In my understanding 3 conv layers and 3 fc layers are already a small model so I did not want to reduce the model size further. Then I need more data - a lazy way to do this if through augmenting the data I already have. Again PyTorch has great tools for this - see my code below. Everytime when I retrieve data from the dataset, it will apply some random adjustments including changing the brightness, contract, random crop and flip so that it appears I have more data. Sure enough the result loop better [here.](.//smaller_model_lr=5e-4_batch=16_dropout_augmentation.png)
```
transform = transforms.Compose(
                    [transforms.ToPILImage(),
                    transforms.Grayscale(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.Resize(size=(256,256)),
                    transforms.RandomCrop(224), # random crop from size 256 to 224
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    ])
```

I could potentially do better by training longer or using greater learning rate. but I estiamted that the test accuracy will be somewhere between 60-70% (vs 20% random guess) which is alright but far from satisfactory. Again, constrained by the small dataset I have, it makes sense to use transfer learning.

### Model training - Transfer Learning with ResNet18
[Ipython Notebook](.//transfer_learning_resnet18.ipynb)
Resnet18 is taking also image of size 224x224, but needs 3 color channels (rgb). The images I downloaded are varying (some are grayscale and some have 4 color channels CMYK). To cope with this, I found a quick trick of add "transforms.Lambda(lambda image: image.convert('RGB'))" to the transforms. There are a few other things in the code that I had to tweak before making it work with resnet18, as well as learning rate tuning, etc. 

It worked amazingly well, and I was able to get ~93% test accuracy for predicting the class (vs 20% random guess). Given that the dataset's quality is relatively poor, I consider this almost a miracle.
<p align="center"> <img src=".//tl_resnet18.png" width="800"/> </p>

I included 100 example test results below. Incorrect predictions are highlighted in red:
<p align="center"> <img src=".//tl_resnet18_examples.jpg" width="800"/> </p>

