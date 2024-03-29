import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import rasterio as rio

from glob import glob
import os

class HLS_US(Dataset):
    def __init__(self, data_dir, file_list, num_classes):
        self.data_dir = data_dir
        self.file_list = file_list
        self.x_dir = os.path.join(data_dir, 'hls')
        self.y_dir = os.path.join(data_dir, 'masks')
        self.num_classes = num_classes

        self.img_norm_cfg = dict(
            means=[
                494.905781,
                815.239594,
                924.335066,
                2968.881459,
                2634.621962,
                1739.579917,
                494.905781,
                815.239594,
                924.335066,
                2968.881459,
                2634.621962,
                1739.579917,
                494.905781,
                815.239594,
                924.335066,
                2968.881459,
                2634.621962,
                1739.579917,
            ],
            stds=[
                284.925432,
                357.84876,
                575.566823,
                896.601013,
                951.900334,
                921.407808,
                284.925432,
                357.84876,
                575.566823,
                896.601013,
                951.900334,
                921.407808,
                284.925432,
                357.84876,
                575.566823,
                896.601013,
                951.900334,
                921.407808,
            ],
        )
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Read random month
        month = np.random.randint(3) * 6
        x = rio.open(os.path.join(self.x_dir, self.file_list[idx] + '_merged.tif')).read(range(month + 1, month + 6 + 1))
        y = rio.open(os.path.join(self.y_dir, self.file_list[idx] + '.mask.tif')).read(1)

        x = torch.Tensor(x)
        y = torch.tensor(y, dtype=torch.int64)

        # TODO: Random flips

        means = self.img_norm_cfg['means'][month: month + 6]
        stds = self.img_norm_cfg['stds'][month: month + 6]
        x = transforms.functional.normalize(x, means, stds)

        y = y - 1
        # y = F.one_hot(y, num_classes=self.num_classes)
        # y = y.permute(2, 0, 1)

        return x, y

if __name__ == '__main__':
    file_list = [os.path.basename(f)[:-11] for f in glob('./data/hls/*')]
    dataset = HLS_US('./data/', file_list, num_classes=13)
    dataloader = DataLoader(dataset, batch_size=2)
    for x, y in dataloader:
        print(x.shape, y.shape)