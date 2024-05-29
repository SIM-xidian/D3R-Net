import numpy as np
from PIL import Image
import torch
def rmse(real,fake):
    real_hr = torch.squeeze(real, 0).transpose(2, 0).cpu().numpy()
    fake_hr = torch.squeeze(fake, 0).transpose(2, 0).cpu().numpy()
    if len(fake_hr.shape) == 3:
        channels = fake_hr.shape[2]
    else:
        channels = 1
        fake_hr = np.reshape(fake_hr, (fake_hr.shape[0], fake_hr.shape[1], 1))
        real_hr = np.reshape(real_hr, (real_hr.shape[0], real_hr.shape[1], 1))
    fake_hr = fake_hr.astype(np.float32)
    real_hr = real_hr.astype(np.float32)

    def single_rmse(img1, img2):
        diff = img1 - img2
        mse = np.mean(np.square(diff))
        return np.sqrt(mse)

    rmse_sum = 0
    for band in range(channels):
        fake_band_img = fake_hr[:, :, band]
        real_band_img = real_hr[:, :, band]
        rmse_sum += single_rmse(fake_band_img, real_band_img)

    rmse = round(rmse_sum, 2)

    return rmse


if __name__ == "__main__":
    image1 = Image.open(r"C:\Users\liuxu\Desktop\img\t\16.png")  # 用PIL中的Image.open打开图像
    image_arr1 = np.array(image1)  # 转化成numpy数组

    image2 = Image.open(r"C:\Users\liuxu\Desktop\img\t\airplane19.tif")  # 用PIL中的Image.open打开图像
    image_arr2 = np.array(image2)  # 转化成numpy数组

    print(rmse(image_arr1,image_arr2))
