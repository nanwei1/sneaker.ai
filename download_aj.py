from download_img import start_download

versions = range(1,24)
prefix = "air_jordan_"
for ver in versions:
    print("==========================================")
    print("Downloading for AJ ", ver)
    start_download(search=prefix+str(ver), num_images=100, verbose=False)
    print("==========================================")
