import os
import argparse
import torch
from torchvision import transforms as tf
from PIL import Image
from D3R.space_diffusion_change import SpacedDiffusion
from D3R.unet_change import UNetModel
from torchvision.transforms import InterpolationMode
from D3R.data import saveImage
import numpy as np
from torchvision import transforms
from torch.autograd import Variable


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
                               # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                               ])}


def restore_rsi(args):
    model = UNetModel(in_channels=3, out_channels=6)
    net = SpacedDiffusion(denoise_fn=model, section=[100])
    device = args.device
    net.model.eval().to(device)
    net.device = torch.device(device)
    net.model.load_state_dict(torch.load(args.model_weight, map_location=device))

    adv_path = args.datapath
    save_path = args.save
    num = 0
    print("start restore.....")
    with torch.no_grad():
        for class_i in sorted(os.listdir(adv_path)):
            if not os.path.exists(os.path.join(save_path, str(class_i))):
                os.mkdir(os.path.join(save_path, str(class_i)))
            for n in sorted(os.listdir(os.path.join(adv_path, class_i))):
                num += 1
                img_path = os.path.join(adv_path, class_i, n)
                x = img_to_tensor(img_path, 256)
                x = Variable(x.cuda())
                x_re = net.d3r_sample_loop(
                    ref_img=x,
                    resizers=True,
                    start_step=args.def_start_teps,
                    output_step=args.def_output_teps,
                )
                saveImage(
                    x=x_re.cpu(),
                    save_dir=os.path.join(save_path, str(class_i)),
                    name=n.split('.')[0],
                )

    print('end restore!')
    print('Sample size:{}'.format(num))


def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument("--use_cuda", default=True,
                        help="Whether use gpu or not")
    parser.add_argument("--device", default='cuda:0',
                        help="Whether use gpu or not")
    # # Path
    parser.add_argument("--dataset", type=str, default='ucm',
                        help='adv-rsi', choices=['ucm', 'aid'])
    parser.add_argument("--datapath", type=str, default='/home/ymhj/桌面/D3RNet/adv_samples/UAE-RS/UCM',
                        help='adv rsi path')
    parser.add_argument("--model_weight", type=str,
                        default='/home/ymhj/桌面/D3RNet/Diffusion_weights/weight/openai-2023-07-07-23-46-33-928071ucm/model110000.pt',
                        help='pretrained weight')
    parser.add_argument("--save", type=str, default='/home/ymhj/桌面/D3RNet/result',
                        help='save restore rsi')


    # Purification hyperparameters in defense
    parser.add_argument("--def_start_teps", type=int, default=16,
                        help='The number of forward steps for each purification step in defense')
    parser.add_argument('--def_output_teps', type=int, default=15,
                        help='The number of denoising steps for each purification step in defense,'
                             'def_output_teps in [0 ~ def_start_teps-1]')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    restore_rsi(args)
