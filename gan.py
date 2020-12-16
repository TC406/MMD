import numpy as np
import os
import skimage.io
import skimage
import skimage.transform
import pandas as pd
from itertools import count

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

import os
from torch import optim


def fetch_lfw_dataset(attrs_name="lfw_attributes.txt",
                      images_name="lfw-deepfunneled",
                      raw_images_name="lfw",
                      use_raw=False,
                      dx=80, dy=80,
                      dimx=45, dimy=45
                      ):  # sad smile

    # download if not exists
    if (not use_raw) and not os.path.exists(images_name):
        print("images not found, donwloading...")
        os.system("wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz -O tmp.tgz")
        print("extracting...")
        os.system("tar xvzf tmp.tgz && rm tmp.tgz")
        print("done")
        assert os.path.exists(images_name)

    if use_raw and not os.path.exists(raw_images_name):
        print("images not found, donwloading...")
        os.system("wget http://vis-www.cs.umass.edu/lfw/lfw.tgz -O tmp.tgz")
        print("extracting...")
        os.system("tar xvzf tmp.tgz && rm tmp.tgz")
        print("done")
        assert os.path.exists(raw_images_name)

    if not os.path.exists(attrs_name):
        print("attributes not found, downloading...")
        os.system("wget http://www.cs.columbia.edu/CAVE/databases/pubfig/download/%s" % attrs_name)
        print("done")

    # read attrs
    df_attrs = pd.read_csv("lfw_attributes.txt", sep='\t', skiprows=1, )
    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])

    # read photos
    dirname = raw_images_name if use_raw else images_name
    photo_ids = []
    for dirpath, dirnames, filenames in os.walk(dirname):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath, fname)
                photo_id = fname[:-4].replace('_', ' ').split()
                person_id = ' '.join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                photo_ids.append({'person': person_id, 'imagenum': photo_number, 'photo_path': fpath})

    photo_ids = pd.DataFrame(photo_ids)

    # mass-merge
    # (photos now have same order as attributes)
    df_attrs['imagenum'] = df_attrs['imagenum'].astype(np.int64)
    df = pd.merge(df_attrs, photo_ids, on=('person', 'imagenum'))

    assert len(df) == len(df_attrs), "lost some data when merging dataframes"

    # image preprocessing
    all_photos = df['photo_path'].apply(lambda img: skimage.io.imread(img)) \
        .apply(lambda img: img[dy:-dy, dx:-dx]) \
        .apply(lambda img: skimage.img_as_ubyte(skimage.transform.resize(img, [dimx, dimy])))

    all_photos = np.stack(all_photos.values).astype('uint8')
    all_attrs = df.drop(["photo_path", "person", "imagenum"], axis=1)

    return all_photos, all_attrs

import numpy as np

# The following line fetches you two datasets: images, usable for autoencoder training and attributes.
# Those attributes will be required for the final part of the assignment (applying smiles), so please keep them in mind
data,attrs = fetch_lfw_dataset(dimx=32, dimy=32)

# Preprocess faces
data = np.float32(data).transpose([0, 3, 1, 2]) / 127.5 - 1.0

IMG_SHAPE = data.shape[1:]

use_cuda = torch.cuda.is_available()


def sample_noise_batch(batch_size):
    noise = torch.randn(batch_size, CODE_SIZE)
    return noise.cuda() if use_cuda else noise.cpu()


class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)

CODE_SIZE = 256

# automatic layer name maker. Don't do this in production :)
ix = ('layer_%i'%i for i in count())

generator = nn.Sequential(
    nn.Linear(CODE_SIZE, 256 * 4 * 4),
    Reshape([-1, 256, 4, 4]),
    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.LeakyReLU(0.2),
    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.LeakyReLU(0.2),
    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.LeakyReLU(0.2),
    nn.Conv2d(32, 3, kernel_size=3, padding=1),
    nn.Tanh()
)

if use_cuda: generator.cuda()

def sample_data_batch(batch_size):
    idxs = np.random.choice(np.arange(data.shape[0]), size=batch_size)
    batch = torch.tensor(data[idxs], dtype=torch.float32)
    return batch.cuda() if use_cuda else batch.cpu()

# a special module that converts [batch, channel, w, h] to [batch, units]
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)

batchSize = 50

def baseline():

    generator = nn.Sequential(
        nn.Linear(CODE_SIZE, 256 * 4 * 4),
        Reshape([-1, 256, 4, 4]),
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 3, kernel_size=3, padding=1),
        nn.Tanh()
    )

    discriminator = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.MaxPool2d(2),
        nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2),
        Reshape([batchSize, 256*16*16]),
        nn.Linear(256*16*16, 1),
        #nn.Sigmoid(),
    )

    if use_cuda: discriminator.cuda()
    if use_cuda: generator.cuda()

    return generator, discriminator

def spectralNorm():

    generator = nn.Sequential(
        spectral_norm(nn.Linear(CODE_SIZE, 256 * 4 * 4)),
        Reshape([-1, 256, 4, 4]),
        spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)),
        nn.LeakyReLU(0.2),
        spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)),
        nn.LeakyReLU(0.2),
        spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)),
        nn.LeakyReLU(0.2),
        spectral_norm(nn.Conv2d(32, 3, kernel_size=3, padding=1)),
        nn.Tanh(),
    )

    discriminator = nn.Sequential(
        spectral_norm(nn.Conv2d(3, 32, kernel_size=3, padding=1)),
        nn.MaxPool2d(2),
        spectral_norm(nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2),
        spectral_norm(nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2),
        spectral_norm(nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1)),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2),
        Reshape([batchSize, 256*16*16]),
        spectral_norm(nn.Linear(256*16*16, 1)),
        #nn.Sigmoid(),
    )

    if use_cuda: discriminator.cuda()
    if use_cuda: generator.cuda()

    return generator, discriminator

def baseline_batchnorm():

    generator = nn.Sequential(
        nn.Linear(CODE_SIZE, 256 * 4 * 4),
        Reshape([-1, 256, 4, 4]),
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.Tanh()
    )

    discriminator = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),
        nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2),
        Reshape([batchSize, 256*16*16]),
        nn.Linear(256*16*16, 1),
        #nn.Sigmoid(),
    )

    if use_cuda: discriminator.cuda()
    if use_cuda: generator.cuda()

    return generator, discriminator

def spectralNorm_batchnorm():

    generator = nn.Sequential(
        spectral_norm(nn.Linear(CODE_SIZE, 256 * 4 * 4)),
        Reshape([-1, 256, 4, 4]),
        spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),
        spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),
        spectral_norm(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2),
        spectral_norm(nn.Conv2d(32, 3, kernel_size=3, padding=1)),
        nn.BatchNorm2d(3),
        nn.Tanh(),
    )

    discriminator = nn.Sequential(
        spectral_norm(nn.Conv2d(3, 32, kernel_size=3, padding=1)),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),
        spectral_norm(nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2),
        spectral_norm(nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2),
        spectral_norm(nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1)),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(2),
        nn.LeakyReLU(0.2),
        Reshape([batchSize, 256*16*16]),
        spectral_norm(nn.Linear(256*16*16, 1)),
        #nn.Sigmoid(),
    )

    if use_cuda: discriminator.cuda()
    if use_cuda: generator.cuda()

    return generator, discriminator

def generatorLoss (z):
    return -discriminator(generator(z)).mean()

def discriminatorLoss (realX, z):
    realPart = (torch.clamp(1 - discriminator(realX), min=0)).mean()
    fakePart = (torch.clamp (1 + discriminator(generator(z)), min=0)).mean()
    return realPart + fakePart

#gradient penalty
from torch.autograd import Variable

def gradPenalty (realX, z):

    realX = Variable(realX, requires_grad=True)
    z = Variable(z, requires_grad=True)

    fakeInput = generator(z)

    outTrue = discriminator(realX)
    outFake = discriminator(fakeInput)

    gradientsTrue = torch.autograd.grad(outputs=outTrue, inputs=realX, grad_outputs=torch.ones(outTrue.size()).cuda(), retain_graph=True)[0]
    gradientsTrue = gradientsTrue.reshape(50, -1) #flattening every sample in the batch
    gradientsFake = torch.autograd.grad(outputs=outFake, inputs=fakeInput, grad_outputs=torch.ones(outFake.size()).cuda(), retain_graph=True)[0]
    gradientsFake = gradientsFake.reshape(50, -1) #flattening every sample in the batch

    gradientsTrue_norm = torch.sqrt(torch.sum(gradientsTrue**2, dim=1))
    gradientsFake_norm = torch.sqrt(torch.sum(gradientsFake**2, dim=1))

    norm = 10* (((gradientsTrue_norm -1 )**2).mean() + ((gradientsFake_norm -1 )**2).mean())
    return norm

# batchSize was set to 50 in the cell with discriminator

# for j in range(6):
# collab interrupted on j=2
for j in range(6):

    if (j==0):
        expName = 'baseline'
        FLAG_gradPenalty=0
        FLAG_specnorm=0
        FLAG_baseline=1 #just to check (no use)
        generator, discriminator = baseline()
        N = 8000
    elif (j==1):
        expName = 'spectralNorm'
        FLAG_gradPenalty=0
        FLAG_specnorm=1
        FLAG_baseline=0 #just to check (no use)
        generator, discriminator = spectralNorm()
        N = 8000
    elif (j==2):
        expName = 'GP'
        FLAG_gradPenalty=1
        FLAG_specnorm=0
        FLAG_baseline=0 #just to check (no use)
        generator, discriminator = baseline()
        N = 8000
        #collab interrupted here
    elif (j==3):
        expName = 'baseline_BN'
        FLAG_gradPenalty=0
        FLAG_specnorm=0
        FLAG_baseline=1 #just to check (no use)
        generator, discriminator = baseline_batchnorm()
        N = 4000
    elif (j==4):
        expName = 'spectralNorm_BN'
        FLAG_gradPenalty=0
        FLAG_specnorm=1
        FLAG_baseline=0 #just to check (no use)
        generator, discriminator = spectralNorm_batchnorm()
        N = 4000
    elif (j==5):
        expName = 'GP_BN'
        FLAG_gradPenalty=1
        FLAG_specnorm=0
        FLAG_baseline=0 #just to check (no use)
        generator, discriminator = baseline_batchnorm()
        N = 4000

    directory = './' + expName
    os.mkdir(directory)

    count = 0
    lr = 1e-4
    if FLAG_specnorm: lr = 4e-4
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0,0.999)) #beta2 set to default
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0,0.999)) #beta2 set to default

    GLosses = []
    DLOsses = []
    while count < N:
        count += 1
        realX = sample_data_batch(50)
        z = sample_noise_batch(50)

        optimizerG.zero_grad()
        GLoss = generatorLoss (z)
        GLoss.backward()
        optimizerG.step()
        GLosses.append(GLoss.detach().cpu().item())

        for i in range (5):
            optimizerD.zero_grad()
            if FLAG_gradPenalty:
                DLoss = discriminatorLoss (realX, z) + gradPenalty (realX, z)
            else:
                DLoss = discriminatorLoss (realX, z)
            DLoss.backward()
            optimizerD.step()
            if i==4:
                DLOsses.append(DLoss.detach().cpu().item())

        if count%1==0:
            out = expName + '; iteration: ' + str(count)
            print(out)
            z = sample_noise_batch(2)
            val_gen = generator(z).data.cpu().numpy()

            model_save_name = expName + '/' + str(count)
            path = F"./{model_save_name}"
            torch.save(generator, path)

            model_save_name = expName + '_D'
            path = F"./{model_save_name}"
            np.save(path, DLOsses)

            model_save_name = expName + '_G'
            path = F"./{model_save_name}"
            np.save(path, GLosses)