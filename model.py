import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from torch import nn
from PIL import Image

from models import fcn, unet

class SegmentationModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.model_name = config.model_name
        self.model_path = config.model_path+'/'+config.model_name

        self.num_classes = config.num_classes
        self.criterion = nn.CrossEntropyLoss()

        if self.model_name=='fcn32s':
            self.net = fcn.FCN32s(num_classes=self.num_classes, in_channels=1)

        elif self.model_name=='fcn16s':
            self.net = fcn.FCN16s(num_classes=self.num_classes, in_channels=1)

        elif self.model_name=='fcn8s':
            self.net = fcn.FCN8s(num_classes=self.num_classes, in_channels=1)

        elif self.model_name=='unet':
            self.net = unet.UNet(num_classes=self.num_classes, in_channels=1)

        #for metric
        self.train_class_iou = [0.] * self.num_classes
        self.train_pixel_acc = 0
        self.valid_class_iou = [0.] * self.num_classes
        self.valid_pixel_acc = 0
        self.test_class_iou = [0.] * self.num_classes
        self.test_pixel_acc = 0

    def forward(self, x):
        return self.net(x)

    def forward_valid(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y, n = batch
        y = y.squeeze(1).long()
        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)
        self.log("train_batch_loss", loss, on_epoch=True, prog_bar=True)

        pred = torch.argmax(y_hat, dim=1)
        batch_size = x.shape[0]

        pred = pred.view(batch_size, -1)
        y = y.view(batch_size, -1)

        self.train_sample_size = np.array(n.cpu()[0])

        self.train_class_iou += self.iou(pred, y, batch_size, self.num_classes)
        self.train_pixel_acc += self.pix_acc(pred, y, batch_size)

        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        avg_iou = np.mean(self.train_class_iou/self.train_sample_size)
        avg_pixel_acc = np.mean(self.train_pixel_acc/self.train_sample_size)


        self.log("train_avg_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("train_avg_iou", avg_iou, on_epoch=True, prog_bar=True)
        self.log("train_avg_pixel_acc", avg_pixel_acc, on_epoch=True, prog_bar=True)


        with open(self.model_path + '/train_loss.txt', "a") as f:
            f.write('{} {} {}'.format(avg_loss, avg_iou, avg_pixel_acc) + '\n')

        self.train_pixel_acc = [0.] * self.num_classes
        self.train_pixel_acc = 0


    def validation_step(self, batch, batch_idx):
        x, y, n = batch
        y = y.squeeze(1).long()
        y_hat = self.forward_valid(x)

        loss = self.criterion(y_hat, y)
        self.log("val_batch_loss", loss, on_epoch=True, prog_bar=True)

        pred = torch.argmax(y_hat, dim=1)
        batch_size = x.shape[0]

        pred = pred.view(batch_size, -1)
        y = y.view(batch_size, -1)

        self.valid_sample_size = np.array(n.cpu()[0])

        self.valid_class_iou += self.iou(pred, y, batch_size, self.num_classes)
        self.valid_pixel_acc += self.pix_acc(pred, y, batch_size)

        return {"val_batch_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_batch_loss'] for x in outputs]).mean()
        avg_iou = np.mean(self.valid_class_iou/self.valid_sample_size)
        avg_pixel_acc = np.mean(self.valid_pixel_acc/self.valid_sample_size)

        self.log("val_avg_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val_avg_iou", avg_iou, on_epoch=True, prog_bar=True)
        self.log("val_avg_pixel_acc", avg_pixel_acc, on_epoch=True, prog_bar=True)

        with open(self.model_path + '/valid_loss.txt', "a") as f:
            f.write('{} {} {}'.format(avg_loss, avg_iou, avg_pixel_acc) + '\n')

        self.valid_class_iou = [0.] * self.num_classes
        self.valid_pixel_acc = 0

        return {"val_avg_loss": avg_loss, "val_avg_iou": avg_iou,
                "val_avg_pixel_acc": avg_pixel_acc}

    def test_step(self, batch, batch_idx):
        x, y, n = batch
        y = y.squeeze(1).long()
        y_hat = self.forward_valid(x)

        loss = self.criterion(y_hat, y)

        pred = torch.argmax(y_hat, dim=1)
        batch_size = x.shape[0]


        ##image save
        if not os.path.exists(self.model_path +"/test_images"):
            os.makedirs(self.model_path +"/test_images")
        pred_images=pred.unsqueeze(1)

        for i in range((pred_images.shape[0])):
            im = Image.fromarray((np.array(pred_images[i][0].cpu())* 255).astype(np.uint8))
            im.save(self.model_path +f"/test_images/test_{i}.jpeg")

        pred = pred.view(batch_size, -1)
        y = y.view(batch_size, -1)

        self.test_sample_size = np.array(n.cpu())[0]

        self.test_class_iou += self.iou(pred, y, batch_size, self.num_classes)
        self.test_pixel_acc += self.pix_acc(pred, y, batch_size)

        return {"test_batch_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_batch_loss'] for x in outputs]).mean()
        avg_iou = np.mean(self.test_class_iou/self.test_sample_size)
        avg_pixel_acc = np.mean(self.test_pixel_acc/self.test_sample_size)


        self.log("test_avg_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_avg_iou", avg_iou, on_epoch=True, prog_bar=True)
        self.log("test_avg_pixel_acc", avg_pixel_acc, on_epoch=True, prog_bar=True)

        with open(self.model_path + '/test_loss.txt', "a") as f:
            f.write('{} {} {}'.format(avg_loss, avg_iou, avg_pixel_acc) + '\n')

        self.test_class_iou = [0.] * self.num_classes
        self.test_pixel_acc = 0

        return {"test_avg_loss": avg_loss, "test_avg_iou": avg_iou,
                "test_avg_pixel_acc": avg_pixel_acc}

    def jaccard(self, y_true, y_pred):
        num = y_true.size(0)
        eps = 1e-7

        y_true_flat = y_true.view(num, -1)
        y_pred_flat = y_pred.view(num, -1)
        intersection = (y_true_flat * y_pred_flat).sum(1)
        union = ((y_true_flat + y_pred_flat) > 0.0).float().sum(1)

        score = (intersection) / (union + eps)
        score = score.sum() / num
        return score

    def dice(self, y_true, y_pred):
        """ Dice a.k.a f1 score for batch of images
        """
        num = y_true.size(0)
        eps = 1e-7

        y_true_flat = y_true.view(num, -1)
        y_pred_flat = y_pred.view(num, -1)
        intersection = (y_true_flat * y_pred_flat).sum(1)

        score = (2 * intersection) / (y_true_flat.sum(1) + y_pred_flat.sum(1) + eps)
        score = score.sum() / num
        return score

    def pix_acc(self, outputs, targets, batch_size):
        """Pixel accuracy
        Args:
            outputs (torch.nn.Tensor): prediction outputs
            targets (torch.nn.Tensor): prediction targets
            batch_size (int): size of minibatch
        """
        acc = 0
        for idx in range(batch_size):
            output = outputs[idx]
            target = targets[idx]
            correct = torch.sum(torch.eq(output, target).long())
            acc += correct / np.prod(np.array(output.shape))

            # acc += correct / np.prod(np.array(output.shape)) / batch_size
        return np.array(acc.item()) * 100

    def iou(self, outputs, targets, batch_size, n_classes):
        """Intersection over union
        Args:
            outputs (torch.nn.Tensor): prediction outputs
            targets (torch.nn.Tensor): prediction targets
            batch_size (int): size of minibatch
            n_classes (int): number of segmentation classes
        """
        eps = 1e-6
        class_iou = np.zeros(n_classes)
        for idx in range(batch_size):
            outputs_cpu = outputs[idx].cpu()
            targets_cpu = targets[idx].cpu()

            for c in range(n_classes):
                i_outputs = np.where(outputs_cpu == c)  # indices of 'c' in output
                i_targets = np.where(targets_cpu == c)  # indices of 'c' in target
                intersection = np.intersect1d(i_outputs, i_targets).size
                union = np.union1d(i_outputs, i_targets).size
                class_iou[c] += (intersection + eps) / (union + eps)

        # class_iou /= batch_size

        return np.array(class_iou) * 100