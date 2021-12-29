import os
import pickle
from PIL import Image
from torch.utils.data import Dataset

class BIRADS(Dataset):

    def __init__(self, root_dir, pkl_file, mode="train", color_channels=3, transforms=None):
        data_pkl = pickle.load(open(pkl_file, "rb"))
        self.data_list = data_pkl[mode]
        self.transforms = transforms
        self.root_dir = root_dir
        self.color_channels = color_channels

    def __len__(self):
        return len(self.data_list)

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

        return img, sample[1]