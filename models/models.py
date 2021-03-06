import numpy as np
import torch.nn as nn
from skimage.measure import compare_ssim as SSIM

from util.metrics import PSNR


class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        # inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, inps, outputs, targets) -> (float, float, np.ndarray):
        psnr = 0
        ssim = 0
        for i in range(len(inps)):
            inp = inps[i:i + 1]
            output = outputs[i:i + 1]
            target = targets[i:i + 1]
            inp = self.tensor2im(inp.data)
            fake = self.tensor2im(output.data)
            real = self.tensor2im(target.data)
            psnr += PSNR(fake, real)
            ssim += SSIM(fake, real, multichannel=True)
            vis_img = np.hstack((inp, fake, real))

        return psnr / len(inps), ssim / len(inps), vis_img


def get_model(model_config):
    return DeblurModel()
