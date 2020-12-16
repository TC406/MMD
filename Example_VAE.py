"""
Authors: 
Eugene Belilovsky
"""

"""This script trains an auto-encoder on the MNIST dataset and keeps track of the lowerbound"""

#python trainmnist.py -s mnist.npy

import VariationalAutoencoder
import numpy as np
import scipy as sp
import time,os
import gzip, pickle,copy,pickle
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from mmd import MMD_3_Sample_Test

import urllib.request
import numpy as np
from PIL import Image


def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")

def download_mnist():
    for v in key_file.values():
       _download(v)

def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data

def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done")

def _change_ont_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """
    Parameters
    ----------
    normalize : Normalize the pixel values
    flatten : Flatten the images as one array
    one_hot_label : Encode the labels as a one-hot array

    Returns
    -------
    (Trainig Image, Training Label), (Test Image, Test Label)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    if one_hot_label:
        dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# Load the MNIST dataset
url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

#
# print("Loading MNIST data")
#Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz

# f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  =  load_mnist(normalize=False, flatten=True)
f.close()
x_train=(x_train>0).astype('float')
x_valid=(x_valid>0).astype('float')
x_test=(x_test>0).astype('float')

data=x_train 
samp_size=1000
        

verbose=True

runs=25
dimZ = 20
HU_decoder = 400
HU_encoder = HU_decoder

dimZ2 = dimZ
HU_decoder2 = HU_decoder
HU_encoder2 = HU_decoder2

batch_size = 100
L = 1
learning_rate = 0.01

sig=0.05

ratio=0.3
#ratios=np.array([0.5,1,2])
t_size2=2000
t_size1=int(t_size2/ratio)
        
set1,set2=train_test_split(list(range(data.shape[0])),train_size=t_size1+t_size2)
data1 = data[set1[0:t_size1],:]
data2 = data[set1[t_size1:t_size2+t_size1],:]
set2=set2[0:samp_size]
data_holdout=data[set2,:]


[N1,dimX] = data1.shape
[N2,dimX] = data2.shape
encoder1 = VariationalAutoencoder.VA(HU_decoder,HU_encoder,dimX,dimZ,batch_size,L,learning_rate,continous=False)
encoder2 = VariationalAutoencoder.VA(HU_decoder2,HU_encoder2,dimX,dimZ2,batch_size,L,learning_rate,continous=False)


print("Creating Theano functions")
encoder1.createGradientFunctions()
encoder2.createGradientFunctions()
print("Initializing weights and biases")
encoder1.initParams()
encoder2.initParams()  

begin = time.time()
maxiter=2000
testlowerbound1=testlowerbound2=-np.Inf
for j in range(maxiter):
    encoder1.iterate(data1)  
    if j%1 == 0:
        oldlower=testlowerbound1
        train_lower1=encoder1.getLowerBound(data1)
        testlowerbound1 = encoder1.getLowerBound(data_holdout)
        if(verbose):
            print(("Encoder 1 Iteration %d| lower bound train = %.2f |lower bound test 1= %.2f"
                  % (j, train_lower1/float(N1),testlowerbound1/samp_size)))
        if(oldlower>=testlowerbound1):
            break  
        best_encoder1=copy.deepcopy(encoder1)
 

encoder1=best_encoder1

for j in range(maxiter):
    encoder2.iterate(data2) 

    if j%1 == 0:
        oldlower=testlowerbound2
        train_lower2=encoder2.getLowerBound(data2)
        testlowerbound2 = encoder2.getLowerBound(data_holdout)
        if(verbose):
            print(("Encoder 2 Iteration %d| lower bound train = %.2f |lower bound test 1= %.2f"
                  % (j, train_lower2/float(N2),testlowerbound2/samp_size)))
        if(oldlower>testlowerbound2):
            break
        best_encoder2=copy.deepcopy(encoder2)
 

encoder2=best_encoder2
end=time.time()
   

samples1=encoder1.sample(N=samp_size)
samples2=encoder2.sample(N=samp_size)

pvalue,tstat,sigma,MMDXY,MMDXZ=MMD_3_Sample_Test(data_holdout,samples1,samples2,computeMMDs=True)
print(("MMD(enc1 samples,real): %.4f MMD(enc2 samples,real): %.4f , pvalue: %.2f"%(MMDXY,MMDXZ,pvalue)))
#Regressions
##Train regression

data1_enc=encoder1.encode(x_valid)
data2_enc=encoder2.encode(x_valid)
test1_enc=encoder1.encode(x_test)
test2_enc=encoder2.encode(x_test)

LogReg_VAE = linear_model.LogisticRegression()
LogReg_VAE.fit(data1_enc,t_valid)
vae1_score = LogReg_VAE.score(test1_enc, t_test)
LogReg_VAE = linear_model.LogisticRegression()
LogReg_VAE.fit(data2_enc,t_valid)
vae2_score = LogReg_VAE.score(test2_enc, t_test)
print(("Accuracy 1:%.2f Accuracy2:%.2f"%(vae1_score,vae2_score)))


