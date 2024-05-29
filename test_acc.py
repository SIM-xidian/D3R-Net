import os
import numpy as np
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import timm
from PIL import Image


def test_acc(target_model='', n_classes=21, img_path='', model_weight=''):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transform = {
        "train": transforms.Compose([transforms.Resize(size=(256, 256)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                     ]),
        "test": transforms.Compose([transforms.Resize(size=(256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ])}

    net = timm.create_model(target_model, pretrained=False, num_classes=n_classes)
    net.to(device)
    net.load_state_dict(torch.load(model_weight))
    net.eval()

    acc = 0.0  # accumulate accurate number / epoch
    num = 0
    for i in os.listdir(img_path):
        label = torch.from_numpy(np.array(int(i)))
        for j in os.listdir(img_path + '/' + i):
            num = num + 1
            data_path = img_path + '/' + i + '/' + j
            img = Image.open(data_path)
            img = data_transform['test'](img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            test_images, test_labels = img.to(device), label.to(device)
            outputs = net(test_images)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels).sum().item()

    test_accurate = acc / num
    print('test_accuracy: %.4f' % (test_accurate))
    return test_accurate


if __name__ == '__main__':
    dataset_path = './result'
    n_classes = 21
    data = 'UCM'
    target_model = 'vgg16'
    model_weight = ''
    _ = test_acc(target_model, n_classes, dataset_path, model_weight)
