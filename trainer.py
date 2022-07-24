import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import transforms
import pytorch_lightning as pl
from model import SegmentationModel

from dataloader import LungDataset
from factory import MakeDataset

def Trainer(config):
    save_path=config.model_path+'/'+str(config.model_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_list, mask_list = MakeDataset(config)

    train = LungDataset(config = config,
                        mode = 'train',
                        img_list = img_list,
                        mask_list = mask_list,
                        transforms=transforms.Resize((config.resize_dim, config.resize_dim)))

    val =  LungDataset(config = config,
                       mode = 'valid',
                       img_list = img_list,
                       mask_list = mask_list,
                       transforms=transforms.Resize((config.resize_dim, config.resize_dim)))

    train_loader = torch.utils.data.DataLoader(train,
                                               num_workers=config.num_workers,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val,
                                             num_workers=config.num_workers,
                                             batch_size=config.batch_size)


    model = SegmentationModel(config)

    checkpoint_callback = ModelCheckpoint(dirpath=save_path,
                                          filename='model-{epoch}-{train_avg_loss:.3f}-{train_avg_iou:.3f}--{train_avg_pixel_acc:.3f}-{val_avg_loss:.3f}-{val_avg_iou:3f}-{val_avg_pixel_acc:.3f}',
                                          save_top_k=5,
                                          verbose=True,
                                          save_weights_only=True,
                                          monitor="val_avg_loss",
                                          mode="min")

    early_stop_callback = EarlyStopping(monitor='val_avg_loss',
                                        patience=30,
                                        mode='min')

    trainer = pl.Trainer(gpus=1,
                         max_epochs=config.epochs,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         num_sanity_val_steps=0,
                         logger=False)

    trainer.fit(model, train_loader, val_loader)