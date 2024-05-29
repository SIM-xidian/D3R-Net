import os
import argparse
import torch
from torchvision import transforms as tf
from PIL import Image
from D3R.space_diffusion_change import SpacedDiffusion
from D3R.unet_change import UNetModel
from torchvision.transforms import InterpolationMode
from D3R.resizer import Resizer
from D3R.data import saveImage
import numpy as np
from torchvision import transforms, datasets
from torch.autograd import Variable
import tools.model as models
from torch import nn
import cv2
import timm
def recreate_image(im_as_var):
    """
        Recreates images from a torch variable
    """
    # reverse_mean = [-0.485, -0.456, -0.406]
    # reverse_std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    reverse_mean = [-0.5, -0.5, -0.5]
    reverse_std = [1 / 0.5, 1 / 0.5, 1 / 0.5]
    recreated_im = im_as_var.data.numpy()[0].copy()
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    # recreated_im = recreated_im[..., ::-1]
    return recreated_im


def img_to_tensor(img_path, size=512):
    IB = InterpolationMode.BICUBIC
    img = Image.open(img_path).convert("RGB")

    tf_method = tf.Compose([
        tf.Resize((size, size), IB),
        tf.ToTensor(),
        tf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tf_method(img).unsqueeze(0)


data_transform = {
    "train": transforms.Compose([transforms.Resize(size=(256, 256)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                 ]),
    "val": transforms.Compose([transforms.Resize(size=(256, 256)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                               # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                               ])}


# def main():
def restore_rsi(args):
    model = UNetModel(in_channels=3, out_channels=6)
    net = SpacedDiffusion(denoise_fn=model, section=[100])
    device = args.device
    net.model.eval().to(device)
    net.device = torch.device(device)
    net.model.load_state_dict(torch.load(args.model_weight, map_location=device))

    adv_path = args.datapath
    save_path = args.save

    pretrained_model = '/home/ymhj/桌面/liuxuehu/zuhui629/weight/AIDMINI/vgg16.pth'
    targetnet = timm.create_model('vgg16', pretrained=False, num_classes=30)
    targetnet.to(device)
    targetnet.load_state_dict(torch.load(pretrained_model))
    targetnet.eval()

    # model_weight = '/home/ymhj/桌面/D3RNet/model_weight/UCM/Pretrain/vgg16/epoch_10_OA_9190.pth'
    # targetnet = models.vgg16(pretrained=False)
    # targetnet.classifier._modules['6'] = nn.Linear(4096, 21)
    # targetnet.to(device)
    # targetnet = nn.DataParallel(targetnet)
    # targetnet.load_state_dict(torch.load(model_weight))
    # targetnet.eval()

    for i in range(3, 25):
        for j in range(1, 8):
            start_step = i + j
            out_step = i
            acc = 0
            with torch.no_grad():
                for class_i in sorted(os.listdir(adv_path)):
                    label = torch.from_numpy(np.array(int(i)))
                    if not os.path.exists(os.path.join(save_path, str(class_i))):
                        os.mkdir(os.path.join(save_path, str(class_i)))
                    for n in sorted(os.listdir(os.path.join(adv_path, class_i))):
                        img_path = os.path.join(adv_path, class_i, n)
                        x = img_to_tensor(img_path, 256)
                        x = Variable(x.cuda())

                        # from thop import profile
                        # from thop import clever_format
                        # input = torch.randn(1, 3, 256, 256).cuda()  # 随机生成一个输入张量，这个尺寸应该与模型输入的尺寸相匹配
                        # flops, params = profile(net.model, inputs=(input, torch.tensor([45]).cuda()))
                        #
                        # flops, params = clever_format([flops, params], '%.5f')
                        #
                        # print(f"运算量：{flops}, 参数量：{params}\n")
                        x_re = net.d3r_sample_loop(
                            ref_img=x,
                            resizers=True,
                            start_step=start_step,
                            output_step=out_step,
                        )
                        saveImage(
                            x=x_re.cpu(),
                            save_dir=os.path.join(save_path, str(class_i)),
                            name=n.split('.')[0],
                        )

            img_path = '/tmpaid'
            acc = 0.0  # accumulate accurate number / epoch
            num = 0
            with torch.no_grad():
                for ii in os.listdir(img_path):
                    label = torch.from_numpy(np.array(int(ii)))
                    for j in os.listdir(img_path + '/' + ii):
                        num = num + 1
                        data_path = img_path + '/' + ii + '/' + j
                        img = Image.open(data_path)
                        img = data_transform['val'](img)
                        # expand batch dimension
                        img = torch.unsqueeze(img, dim=0)

                        test_images, test_labels = img.to(device), label.to(device)
                        outputs = targetnet(test_images)
                        predict_y = torch.max(outputs, dim=1)[1]
                        acc += torch.eq(predict_y, test_labels).sum().item()

            test_accurate = acc / num
            print('test_accuracy: %.4f' % (test_accurate))

            with open('examplesaid.txt', 'a+') as f:
                f.write("out_step:{} start_step:{}  |  acc:{}\n".format(out_step, start_step, '%.4f'%test_accurate))


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument("--use_cuda", default=True,
                        help="Whether use gpu or not")
    parser.add_argument("--device", default='cuda:0',
                        help="Whether use gpu or not")
    # Path
    parser.add_argument("--dataset", type=str, default='aid',
                        help='adv-rsi', choices=['ucm', 'aid'])
    parser.add_argument("--datapath", type=str, default='/home/ymhj/桌面/liuxuehu/get_adv/aidmini/val',
                        help='adv rsi path')
    parser.add_argument("--model_weight", type=str,
                        default='/home/ymhj/桌面/liuxuehu/Diffusion_weight/weight//openai-2023-07-08-22-19-00-003501AID/model100000.pt',
                        help='pretrained weight')
    parser.add_argument("--save", type=str, default='/home/ymhj/桌面/D3RNet/tmpaid',
                        help='save restore rsi')
    parser.add_argument('--batch_size', type=int, default=1)

    # Purification hyperparameters in defense
    parser.add_argument("--def_start_teps", type=int, default=15,
                        help='The number of forward steps for each purification step in defense')
    parser.add_argument('--def_output_teps', type=int, default=12,
                        help='The number of denoising steps for each purification step in defense')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    restore_rsi(args)
