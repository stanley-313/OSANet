import time
import argparse
import scipy.misc
import torch.backends.cudnn as cudnn
from utils.utils import *
from OSANet import Net
from tqdm import tqdm
import h5py
import numpy as np
from torchvision.transforms import ToTensor
import torch
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='OSANet')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")
    parser.add_argument('--testset_dir', type=str, default='./data_for_test/')
    parser.add_argument('--testdata', type=str, default='SR_5x5_4x/')
    parser.add_argument("--patchsize", type=int, default=32, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--stride", type=int, default=16,
                        help="The stride between two test patches is set to patchsize/2")
    parser.add_argument('--channels', type=int, default=60, help='channels')
    parser.add_argument('--model_path', type=str, default='./pth/OSANet_4xSR_5x5.pth.tar')
    # parser.add_argument('--model_path', type=str, default='../NTIRE23_LFSR_DistgEPIT/checkpoints/Exp80.DistgEPITv6.B4.L5e-5.E100.P128_32mp.S5.EMA0.999.FT.pth')
    parser.add_argument('--save_path', type=str, default='./Results/')

    return parser.parse_args()


def test(cfg, test_Names, test_loaders):
    net = Net(cfg.angRes, cfg.upscale_factor, cfg.channels)
    net.to(cfg.device)
    cudnn.benchmark = True

    if os.path.isfile(cfg.model_path):
        model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
        try:
            net.load_state_dict(model['state_dict'])
        except:
            net.load_state_dict({k.replace('module.netG.', ''): v for k, v in model['state_dict'].items()})
    else:
        print("=> no model found at '{}'".format(cfg.model_path))

    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            outLF, psnr_epoch_test, ssim_epoch_test = inference(test_loader, test_name, net)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (
                test_name, psnr_epoch_test, ssim_epoch_test))
            pass
        print(time.ctime()[4:-5] + ' Average_Result: Average_PSNR---%.6f, Average_SSIM---%.6f' % (
            (sum(psnr_testset) / len(psnr_testset)), (sum(ssim_testset) / len(ssim_testset))))
        pass


def inference(test_loader, test_name, net):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in tqdm((enumerate(test_loader)), total=len(test_loader), ncols=70):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angRes, w*angRes
        label = label.squeeze()

        uh, vw = data.shape
        h0, w0 = uh // cfg.angRes, vw // cfg.angRes
        subLFin = LFdivide(data, cfg.angRes, cfg.patchsize, cfg.stride)  # numU, numV, h*angRes, w*angRes
        numU, numV, H, W = subLFin.shape
        subLFout = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize * cfg.upscale_factor,
                               cfg.angRes * cfg.patchsize * cfg.upscale_factor)

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(
                    0)  # patchsize 128 tmp (1,1,640,640)  patchsize 32 tmp (1,1,160,160)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    out = net(tmp.to(cfg.device))
                    subLFout[u, v, :, :] = out.squeeze()

        outLF = LFintegrate(subLFout, cfg.angRes, cfg.patchsize * cfg.upscale_factor, cfg.stride * cfg.upscale_factor,
                            h0 * cfg.upscale_factor, w0 * cfg.upscale_factor)

        psnr, ssim = cal_metrics(label, outLF, cfg.angRes)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        save_path = cfg.save_path + '/' + cfg.model_name + '/' + cfg.testdata

        isExists = os.path.exists(save_path + test_name)
        if not (isExists):
            os.makedirs(save_path + test_name)

        from scipy import io
        scipy.io.savemat(save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
                         {'LF': outLF.numpy()})
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return outLF, psnr_epoch_test, ssim_epoch_test

def main(cfg):
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    test(cfg, test_Names, test_Loaders)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
