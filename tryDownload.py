# from download_img import start_download
from download_img2 import start_download

versions = [8]
prefix = "air_jordan_"
for ver in versions:
    print("==========================================")
    print("Downloading for AJ ", ver)
    start_download(search=prefix+str(ver), num_images=200, verbose=False)
    print("==========================================")
