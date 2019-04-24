This is a toy personal project in which I am trying to develop an algorithm that can tell user the what sneaker it is given an image. Work in progress.

## Data Scraping: (2019/04/13)
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
I started off by using only 5 classes (Air Jordan 1 - 5) instead of doing the full 23 classes to get a quick start. If it turns out that I need to download more data or something else at least I can know early.

### Pre-processing of data
- Delete corrupt files
- Delete random images (posters, drawings, fish pic, pics with DJ Khaled holding the shoes...)
- Generate pandas dataframe recording filename and labels - this step is achieved by running "data_prep.ipynb." Here I randomly split the data set into training and test (80% and 20%) respectively. It is better to use a validation set to do hyperparameter tuning, model selection etc. while holding out the test dataset until the very end, but since this is a toy project and I really do not have too much data to start with so this was ignored.

### Model training - CNN from Scratch
[Ipython Notebook](.//prototype_5/cnn.ipynb)
First thing is to make sure the ingested data are of the same size. The images downloaded have a wide variety of types (grayscale, rgb, etc.) and sizes. PyTorch provides tools for this in the "transforms" package. An example would be to use the following transformation
```
transforms.Compose([transforms.ToPILImage(),  # allowing for grayscale, resize, etc but need ToSensor() in the end
                    transforms.Grayscale(),   # convert all images into grayscale 
                    transforms.Resize(size=224),  # resize to 224 by 224
                    transforms.ToTensor()])
```

I initially attempted to train a CNN from scratch with images transformed to grayscale with 224x224 size. Tried a few different setups (architecture, learning rate, batch size, dropout, transformation, etc). A quick recap of the process (see cnn.ipynb for code):

- Started with a bigger net with 5 conv layers with 3 fc layers. Training loss was reducing while test loss when up. It is clear that there is overfitting. [Result here.](.//prototype_5/lr=1e-4_batch=16.png)
- Added dropout at the fc layers to reduce overfitting. Result was barely better. I think the model is too big for dataset this small. [Result here.](.//prototype_5/lr=1e-4_batch=16_dropout.png)
- Recuced to conv layers to 3. Overfitting was greatly reduced, but training was slow. [Result here.](.//prototype_5/smaller_model_lr=1e-4_batch=16_dropout.png)
- Increased learning rate from 1e-4 to 5e-4. Traning is faster now but overfitting appears again. In fact in the previous attemp there is also overfitting but it started toward the end of training so I did not notice. [Result here.](.//prototype_5/smaller_model_lr=5e-4_batch=16_dropout.png)
- How can I further reduce overfitting? In my understanding 3 conv layers and 3 fc layers are already a small model so I did not want to reduce the model size further. Then I need more data - a lazy way to do this if through augmenting the data I already have. Again PyTorch has great tools for this - see my code below. Everytime when I retrieve data from the dataset, it will apply some random adjustments including changing the brightness, contract, random crop and flip so that it appears I have more data. Sure enough the result loop better [here.](.//prototype_5/smaller_model_lr=5e-4_batch=16_dropout_augmentation.png)
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

I included 100 example test results below. Incorrect predictions are highlighted in red:
<p align="center"> <img src=".//prototype_5/cnn_examples.jpg" width="800"/> </p>

### Model training - Transfer Learning with ResNet18
[Ipython Notebook](.//prototype_5/transfer_learning_resnet18.ipynb)
Resnet18 is taking also image of size 224x224, but needs 3 color channels (rgb). The images I downloaded are varying (some are grayscale and some have 4 color channels CMYK). To cope with this, I found a quick trick of add "transforms.Lambda(lambda image: image.convert('RGB'))" to the transforms. There are a few other things in the code that I had to tweak before making it work with resnet18, as well as learning rate tuning, etc. 

It worked amazingly well, and I was able to get ~95% test accuracy for predicting the class (vs 20% random guess). Given that the dataset's quality is relatively poor, I consider this almost a miracle.
<p align="center"> <img src=".//prototype_5/tl_resnet18.png" width="800"/> </p>

I included 100 example test results below. Incorrect predictions are highlighted in red:
<p align="center"> <img src=".//prototype_5/tl_resnet18_examples.jpg" width="800"/> </p>


## Prototyping (2019/04/15)
**update 2019/04/15:**
I trained the resnet18 again on AJ1 to 10 while screening the rest of the images. Again the results were promisiing with over 90% test accuracy:
<p align="center"> <img src=".//prototype_10/tl_resnet18.png" width="800"/> </p>

I included 100 example test results below. Incorrect predictions are highlighted in red:
<p align="center"> <img src=".//prototype_10/tl_resnet18_examples.jpg" width="800"/> </p>

## Full Model (2019/04/15)
The model was trained on AJ1 to AJ23 for 25 epochs. Again the test accuracy was ~92% in the end. See training history, confusion matrix, and examples below:
<p align="center"> <img src=".//full_model/tl_resnet18.png" width="800"/> </p>
<p align="center"> <img src=".//full_model/tl_resnet18_confusion_matrix.jpg" width="800"/> </p>
<p align="center"> <img src=".//full_model/tl_resnet18_examples.jpg" width="800"/> </p>


## Tensorflow 2.0 Implementation (2019/04/16-2019/04/21)

**Some issues that I encountered with Tensorflow 2.0:**
- cuDNN error, will not start training. *"Error : Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above."* Do the following to prevent tf from using the full GPU memory somehow does the trick. Read about this issue [here](https://github.com/tensorflow/tensorflow/issues/24828).
```
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()
```
- Program killed during training when mobilenetv2 is set as trainable: Caused by RAM running out of space. Adjusted the dataloader buffer to a smaller size to save up RAM.
- Lots of warnings during training that says *"libpng warning: iCCP: known incorrect sRGB profile"*: found a solution [here](https://stackoverflow.com/questions/22745076/libpng-warning-iccp-known-incorrect-srgb-profile) to remove the invalid iCCP chunk. Need pngcrush and run the following in screened_data. It takes a while due to the number of images in the folder.
```
find . -type f -iname '*.png' -exec pngcrush -ow -rem allb -reduce {} \;
```
- After training with the jupyter notebook for several times, I got this error *"TensorBoard could not bind to port 6006."* I believe this is caused by tensorboard not closed properly in Jupyter notebook as one would do in the command line. One work-around that I used for this error is to manualy kill the previous tensorboard processes (replace PID with the process ID found by the first command):
```
ps -ef | grep tensorboard
kill -9 PID
```

Tensorflow does not have ResNet18 ready to use so I am using MobileNetV2 which is comparable in terms of size and performance:
<p align="center"> <img src=".//pretrained_cnns.jpeg" width="600"/> </p>

One can add multiple extra layers after the mobilenet to get better performance, but in my experience it is better to make the mobilenet trainable so the feature extractors can also be adjusted during training. This comes with the cost of more computation as we need to keep track of a lot more gradient vectors and perform updates with them.

Example TensorBoard history:
 <p align="center"> <img src=".//tensorboard_example.png" width="600"/> </p>

Ater adjusting the model and its hyper parameters (mobilenet trainable/non-trainable, learning rate, dropout, regularization, fc architectures, etc.) I was able to achieve around ~90% accuracy. This is pretty good in my opinion, given the poor data quality plus the fact that random rotation is not yet available in Tensorflow2.0 (tf.contrib is removed). 

It is interesting to note that the model was able to catch mistakes in training labels. See image below for example. The obvious AJ1 was labeled as AJ19 but the model was clever enough to recognize it. I suspect there are more of such instances, but I am not proficient (or patient for that matter) to sort through all 23 versions or nearly 7000 images so...
 <p align="center"> <img src=".//wrong_labels.png" width="400"/> </p>

Never mind - after tuning the model and training for countless times, the test accuracy was only able to reach the high 80s. I have tried tuning the learning rate, changing the fc layers, toggling the "trainable" of the mobilenet, adjusting dropout and regularization, etc. because one of my previous suspicion was overfitting (high training accuracy but relatively low test accuracy). It is true that there is a some slight overfitting, but the main culprit was data quality. I used a trained model to display images that got wrongly classified, and manully went over them. Sure enough - there were wrong labels, random images not related to shoes, really bad angle shots that does not give enough information. I spent some time sorting through those.

Another thing that bothered me a lot was training time. Essentially the training step involves loading the full-size images and transform them (with some random augmentation) into 224x224 that the model can take. This was fine for a while, but as I was tuning the model, this became a headache. To this end I used *skimage* to convert all images to 256x256 with [this script](.//image_preprocessing.ipynb). This almost doubled my training speed!

Because tensorflow2.0 removed *tf.contrib*, there was no built-in utility that I could use to randomly rotate the images for data augmentation. I later found that they have migrated to [*tensorflow addons*](https://github.com/tensorflow/addons). However, the random rotation almost doubled training time which I guess was because it was not optimized with the rest of tf2.0. With my input image size shrunken, I think they can be back in use. Let's see...
 
## Local Deployment Using flaskSaaS (2019/04/22 - )
Before trying out cloud-based web app hosting services, I think it is a great learning experience to try do host the web app locally to see how things play out. This is my first experience with backend developments.

Luckily there are tools and templates readily available that can supercharge the start. I was inspired by Siraj Raval's [AI Startup Protoype](https://www.youtube.com/watch?v=NzmoPqte4V4) which used this [flaskSaaS template](https://github.com/alectrocute/flaskSaaS) by ALec Armbruster. Many things have already been taken care of, but there are still many things to sort out before being able to upload an image of random size on a web and process it using the neural net. I had to quickly get some html and css learning on the fly but it was not bad overall given that a minimum viable product was made within a day or two.

As a side note, a difficuty I faced was that the previous saved tensorflow models could not be loaded properly [(see here)](https://github.com/tensorflow/tensorflow/issues/26814). A solution is to update my current tf2.0 preview to a dev version which I am not very keen about. I instead retrained the model and only saved the weights instead of the whole model. See the ipython notebook for details.

A quick demo (with a not-so-well-trained neural net) is in [this folder](.//demo_vids). The model was trained for fewer epochs because I wanted to see the web app working first.

I've retrained the network and uploaded a [demo video](https://www.youtube.com/watch?v=hmd7DfU2o3I). (For convenience a account was registered prior with username "johndoe@johndoe.com" and password "johndoe") There was an error message around 1:58; this was because the test image has 4 channels (RGBY) instead of 3 channels (RGB) that the script was able to handel. Inference takes some time in this demo, because for this MVP, I coded it such that the model has to be initialized every time the user uploads an image. This can be easily optimized for production so that the model is loaded when the server starts.
