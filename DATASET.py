
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os


class CMDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self,
        partition='train',
        data_name='data_name'
    ):
        self.data_name = data_name
        self.partition = partition
        self.open_data()

    def open_data(self):
        if self.data_name.lower() == 'mirflickr25k_deep':
            self.imgs, self.texts, self.labels = MIRFlickr25K_fea(
                self.partition)
        elif self.data_name.lower() == 'iaprt_tc12':
            self.imgs, self.texts, self.labels = IAPR_fea(self.partition)
        elif self.data_name.lower() == 'nus_wide_deep':
            self.imgs, self.texts, self.labels = NUSWIDE_fea(self.partition)
        elif self.data_name.lower() == 'mscoco_deep':
            self.imgs, self.texts, self.labels = MSCOCO_fea(self.partition)
        self.length = self.labels.shape[0]
        self.text_dim = self.texts.shape[1]
        self.label_dim = self.labels.shape[1]

    def __getitem__(self, index):
        image = self.imgs[index]
        text = self.texts[index]
        label = self.labels[index]
        return image, text, label

    def __len__(self):
        return self.length


def MIRFlickr25K_fea(partition):
    root = '/media/hdd4/liy/data/MIRFLICKR25K'
    data_img = sio.loadmat(os.path.join(
        root, 'mirflickr25k-iall-vgg-rand.mat'))['XAll']
    data_txt = sio.loadmat(os.path.join(
        root, 'mirflickr25k-yall-rand.mat'))['YAll']
    labels = sio.loadmat(os.path.join(
        root, 'mirflickr25k-lall-rand.mat'))['LAll']
    test_size = 2000
    train_size = 10000
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size:
            :], data_txt[-test_size::], labels[-test_size::]
    elif 'train' in partition.lower():
        data_img, data_txt, labels = data_img[0:
                                              train_size], data_txt[0: train_size], labels[0: train_size]
    else:
        data_img, data_txt, labels = data_img[0: -
                                              test_size], data_txt[0: -test_size], labels[0: -test_size]

    return data_img, data_txt, labels


def IAPR_fea(partition):
    root = './data/IAPR-TC12/'
    file_path = os.path.join(root, 'iapr-tc12-rand.mat')
    data = sio.loadmat(file_path)

    valid_img = data['VDatabase'].astype('float32')
    valid_txt = data['YDatabase'].astype('float32')
    valid_labels = data['databaseL']

    test_img = data['VTest'].astype('float32')
    test_txt = data['YTest'].astype('float32')
    test_labels = data['testL']

    data_img, data_txt, labels = np.concatenate([valid_img, test_img]), np.concatenate(
        [valid_txt, test_txt]), np.concatenate([valid_labels, test_labels])

    test_size = 2000
    train_size = 10000
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size:
            :], data_txt[-test_size::], labels[-test_size::]
    elif 'train' in partition.lower():
        data_img, data_txt, labels = data_img[0:
                                              train_size], data_txt[0: train_size], labels[0: train_size]
    else:
        data_img, data_txt, labels = data_img[0: -
                                              test_size], data_txt[0: -test_size], labels[0: -test_size]
    return data_img, data_txt, labels


def NUSWIDE_fea(partition):
    root = './data/NUS-WIDE-TC21/'

    data_img = sio.loadmat(root + 'nus-wide-tc21-xall-vgg.mat')['XAll']
    data_txt = sio.loadmat(root + 'nus-wide-tc21-yall.mat')['YAll'][()]
    labels = sio.loadmat(root + 'nus-wide-tc21-lall.mat')['LAll']
    test_size = 2000
    train_size = 10500

    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size:
            :], data_txt[-test_size::], labels[-test_size::]
    elif 'train' in partition.lower():
        data_img, data_txt, labels = data_img[0:
                                              train_size], data_txt[0: train_size], labels[0: train_size]
    else:
        data_img, data_txt, labels = data_img[0: -
                                              test_size], data_txt[0: -test_size], labels[0: -test_size]
    return data_img, data_txt, labels


def MSCOCO_fea(partition):
    root = './data/MSCOCO/'
    import h5py

    path = root + 'MSCOCO_deep_doc2vec_data.h5py'
    data = h5py.File(path)
    data_img = np.concatenate(
        [data['train_imgs_deep'][()], data['test_imgs_deep'][()]], axis=0)
    data_txt = np.concatenate(
        [data['train_text'][()], data['test_text'][()]], axis=0)
    labels = np.concatenate(
        [data['train_imgs_labels'][()], data['test_imgs_labels'][()]], axis=0)
    test_size = 5000
    train_size = 10000

    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size:
            :], data_txt[-test_size::], labels[-test_size::]
    elif 'train' in partition.lower():
        data_img, data_txt, labels = data_img[0:
                                              train_size], data_txt[0: train_size], labels[0: train_size]
    else:
        data_img, data_txt, labels = data_img[0: -
                                              test_size], data_txt[0: -test_size], labels[0: -test_size]
    return data_img, data_txt, labels
