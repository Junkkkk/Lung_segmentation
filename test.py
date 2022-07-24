import argparse
import torch
from glob import glob
from torchvision import transforms
import pytorch_lightning as pl
from model import SegmentationModel

from dataloader import LungDataset
from factory import MakeDataset

def Inference(config):
    img_list, mask_list = MakeDataset(config)

    test = LungDataset(config = config,
                        mode = 'test',
                        img_list = img_list,
                        mask_list = mask_list,
                        transforms=transforms.Resize((config.resize_dim, config.resize_dim)))

    test_loader = torch.utils.data.DataLoader(test,
                                             num_workers=config.num_workers,
                                             batch_size=config.batch_size,
                                              shuffle=False)

    final_model = sorted(glob(config.model_path+'/'+str(config.model_name)+'/*.ckpt'))[-1]

    model = SegmentationModel(config)
    model = model.load_from_checkpoint(final_model, config=config)

    trainer = pl.Trainer(gpus=1, logger=False)
    res=trainer.test(model, test_loader, verbose=False)

    print(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--img_path', type=str, default='/tf/storage/nrcdbox/data/lung_xray/dataset/images')
    parser.add_argument('--mask_path', type=str, default='/tf/storage/nrcdbox/data/lung_xray/dataset/masks')

    parser.add_argument('--model_path', type=str, default='/tf/storage/result/2d_cnn/lung_seg')
    parser.add_argument('--model_name', type=str, default='unet')

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--resize_dim', type=int, default=512)

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=7)

    config = parser.parse_args()

    model_names = ['fcn32s', 'fcn16s', 'fcn8s','unet']
    for model_name in model_names:
        config.model_name = model_name
        Inference(config)