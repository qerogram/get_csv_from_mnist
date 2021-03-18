# Author : qerogram
import numpy as np
import pandas as pd

import os, struct
import requests, hashlib, gzip

def getMd5(data) :    
    hash = hashlib.new("md5")
    hash.update(data)
    return hash.hexdigest()

def fileDownload(url) :
    filename = getMd5(os.urandom(16))
    res = requests.get(url)
    res.raw.decode_content = True

    f = open(filename, 'wb')
    f.write(res.content)
    f.close()

    os.makedirs(filename + "_", exist_ok=True)

    with open(filename +"_/" + filename, "wb") as out_f, gzip.GzipFile(filename) as zip_f:
        out_f.write(zip_f.read())
    
    return filename

def removeFile(filename) : 
    os.remove(filename + "_/" + filename)
    os.rmdir(filename + "_")
    os.remove(filename)

def download_mnist(method):
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    download_link = {
        "train" : ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
        "test" : ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    
    }
    images_path, labels_path = download_link[method]

    label_file_name = fileDownload(base_url + labels_path)

    with open(label_file_name + "_/" + label_file_name,'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    
    removeFile(label_file_name)

    image_file_name = fileDownload(base_url + images_path)

    with open(image_file_name + "_/" + image_file_name,'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        print(f"count of row = {num}, count of column = {rows * cols}")
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), rows * cols)
    
    removeFile(image_file_name)

    return images, labels


if __name__ == '__main__':
    X_train, y_train = download_mnist('train')

    x_df = pd.DataFrame(X_train)
    x_df.to_csv("train_dataset.csv", index=False)

    y_df = pd.DataFrame(y_train)
    y_df.to_csv("train_label.csv", index=False)

    X_test, y_test = download_mnist('test')

    x_df = pd.DataFrame(X_test)
    x_df.to_csv("test_dataset.csv", index=False)

    y_df = pd.DataFrame(y_test)
    y_df.to_csv("test_label.csv", index=False)