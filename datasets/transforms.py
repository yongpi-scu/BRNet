import numpy as np
from torchvision import transforms

class Multi_Input(object):
    def __init__(self, size):
        self.size = size
        self.tf_1 = transforms.Resize(size)
        self.tf_2 = transforms.CenterCrop([size, size])
        self.tf_3 = transforms.Resize([size, size])
        
    def __call__(self,img):
        img = self.tf_1(img).convert("L")
        center_img = np.array(self.tf_2(img))
        resized_img = np.array(self.tf_3(img))
        img = np.array(img)
        left_img = img[0:self.size,0:self.size]
        right_img = img[-self.size:,-self.size:]
        img = np.stack([left_img,center_img,right_img,resized_img],axis=-1)
        return img