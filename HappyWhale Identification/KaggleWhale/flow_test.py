from custom_data import all_custom_setting
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# train_loader, cutmix = all_custom_setting()

trainset = all_custom_setting()
train_loader = DataLoader(trainset, batch_size=32, shuffle= True, pin_memory= True)
iter_data = iter(train_loader)
data, target = next(iter_data)
# data_aug, target = cutmix(data, target, 1.)

for i in range(3):
    f, axarr = plt.subplots(1,4)
    for p in range(0,3,2):
        idx = np.random.randint(0, len(data))
        img_org = data[idx]
        new_img = data[idx]
        axarr[p].imshow(img_org.permute(1,2,0))
        axarr[p+1].imshow(new_img.permute(1,2,0))
        axarr[p].set_title('cutmix image')
        axarr[p+1].set_title('cutmix image')
        axarr[p].axis('off')
        axarr[p+1].axis('off')

plt.show()