from trainer import Trainer
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--img_path', type=str, default='/tf/storage/nrcdbox/data/lung_xray/dataset/images')
parser.add_argument('--mask_path', type=str, default='/tf/storage/nrcdbox/data/lung_xray/dataset/masks')

parser.add_argument('--model_path', type=str, default='/tf/storage/result/2d_cnn/lung_seg')
parser.add_argument('--model_name', type=str, default='unet')

parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--resize_dim', type=int, default=512)

parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--gpu', type=int, default=7)

config = parser.parse_args()
print(config)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(config.gpu)
    Trainer(config)