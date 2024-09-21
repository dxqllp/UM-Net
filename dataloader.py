import os
from utils.transform import *
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2


def get_mean_and_std(img):
    x_mean, x_std = cv2.meanStdDev(img)  
    x_mean = np.hstack(np.around(x_mean, 2)) 
    x_std = np.hstack(np.around(x_std, 2))

    return x_mean, x_std


class Polyp_Dataset(Dataset):
    def __init__(self, root, data_dir, mode, transform=None):
        super(Polyp_Dataset, self).__init__()
        self.mode = mode
        self.imglist = []
        self.gtlist = []
        self.color = []

        data_path = os.path.join(root, data_dir)
        datalist = os.listdir(data_path + '/images')
        for data in datalist:
            self.imglist.append(os.path.join(data_path + '/images', data))
            self.gtlist.append(os.path.join(data_path + '/masks', data))

        transfer_data = os.listdir(root + '/color_transfer')
        for name in transfer_data:
            self.color.append(os.path.join(root + '/color_transfer', name))

        if transform is None:
            if mode == 'train':
                transform = transforms.Compose([
                    Resize((320, 320)),  # (288, 384), (320, 320), (352, 352)
                    RandomHorizontalFlip(0.5),
                    RandomVerticalFlip(0.5),
                    RandomRotation(90),
                    RandomZoom((0.9, 1.1)),
                    RandomCrop((224, 224)),
                    ToTensor(),
                    Normalization()])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                    Resize((320, 320)),
                    ToTensor(),
                    Normalization()])

        self.transform = transform

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.imglist[index]
            gt_path = self.gtlist[index]
            file_name = img_path.split('\\')[-1] 

            img1 = cv2.imread(img_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
            img1_mean, img1_std = get_mean_and_std(img1)

            color_path = self.color[(random.randint(0, len(self.color))) % len(self.color)]
            img2 = cv2.imread(color_path)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
            img2_mean, img2_std = get_mean_and_std(img2)

            img3 = (img1 - img1_mean) / img1_std * img2_std + img2_mean
            np.putmask(img3, img3 > 255, 255)
            np.putmask(img3, img3 < 0, 0)
            image = cv2.cvtColor(cv2.convertScaleAbs(img3), cv2.COLOR_LAB2RGB)
            image = Image.fromarray(image)
            gt = Image.open(gt_path).convert('L')

            data = {'image': image, 'label': gt}
            data = self.transform(data)

            return data, file_name

        elif self.mode == 'valid' or self.mode == 'test':
            img_path = self.imglist[index]
            gt_path = self.gtlist[index]
            file_name = img_path.split('\\')[-1]

            img = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')

            data = {'image': img, 'label': gt}
            data = self.transform(data)

            return data, file_name

    def __len__(self):
        return len(self.imglist)





