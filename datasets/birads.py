import os
import random
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class BIRADS(Dataset):

    def __init__(self, root_dir, pkl_file, mode="train", color_channels=4, transforms=None, oversample=False,
                channel_order_classes=24, bias_original_order=0.6):
        data_pkl = pickle.load(open(pkl_file, "rb"))
        if mode=="train" and oversample:
            self.data_list = self.__over_sample__(data_pkl[mode])
        else:
            self.data_list = data_pkl[mode] # data list [img_path, label]
        self.transforms = transforms
        self.root_dir = root_dir
        self.color_channels = color_channels
        self.channel_order_classes = channel_order_classes
        self.bias_original_order = bias_original_order
        self.permutations = self.__retrieve_permutations()
        self.mode = mode

    def __len__(self):
        return len(self.data_list)

    def __over_sample__(self, data_list):
        samples = {}
        for sample in data_list:
            if sample[1] in samples.keys():
                samples[sample[1]].append(sample)
            else:
                samples[sample[1]] = [sample]
        max_num = 0
        for key in samples:
            print(key, len(samples[key]))
            if len(samples[key])>max_num:
                max_num = len(samples[key])

        oversample_datalist = []
        for key in samples:
            category = []
            random.shuffle(samples[key])
            while len(category)<max_num:
                category.extend(samples[key])
            category = category[:max_num]
            # print(key, len(category))
            oversample_datalist.extend(category)
        return oversample_datalist

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        if self.color_channels==1:
            img = Image.open(os.path.join(self.root_dir, sample[0])).convert("L")
        else:
            img = Image.open(os.path.join(self.root_dir, sample[0]))

        if self.transforms is not None:
            if isinstance(self.transforms, list):
                for trans in self.transforms:
                    img = trans(img)
            else:
                img = self.transforms(img)
        if self.mode=="train":
            img, order = self.__get_channel_order(img)
            return img, order, sample[1]
        else:
            return img, 0, sample[1]

    def __get_channel_order(self, img):
        order = np.random.randint(len(self.permutations))
        if self.bias_original_order:
            if self.bias_original_order > random.random():
                order = 0
        
        if order == 0:
            return img, order
        else:
            return img[self.permutations[order]], order

    def __retrieve_permutations(self):
        all_perm = np.load('%s/channel_order_permutation/permutations_%d.npy' % (os.path.dirname(__file__), self.channel_order_classes))
        return all_perm