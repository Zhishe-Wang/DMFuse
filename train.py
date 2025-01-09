import torch
import time
import utils
from utils import make_floor
import pytorch_msssim
import loss
from Net import net
import os
import random
import numpy as np
from args_fusion import args1
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.autograd import Variable
import scipy.io as scio

EPSILON = 1e-5
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    original_ir_path = args1.data_ir_set
    original_vi_path = args1.data_vi_set
    batch_size = args1.batch_size
    train(original_ir_path, original_vi_path, batch_size)

def train(original_ir_path, original_vi_path, batch_size):

    models_save_path = make_floor(os.getcwd(), args1.save_model_dir)
    print(models_save_path)
    DF_model = net()
    g_content_criterion = loss.g_content_loss().cuda()
    if args1.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args1.resume))
        DF_model.load_state_dict(torch.load(args1.resume))

    optimizer = Adam(DF_model.parameters(), args1.lr)
    grad = loss.grad().cuda()

    if args1.cuda:
        DF_model.cuda()
        grad.cuda()

    tbar = trange(args1.epochs) # 主要目的在于显示进度条
    print('Start training.....')

    count_loss = 0
    Loss_SSIM = []
    Loss_Texture =[]
    Loss_Indensity = []
    Loss_all = []

    for e in tbar:
        print('Epoch %d.....' % e)
        # torch.cuda.empty_cache()  # 释放显存
        image_set_ir, image_set_vi, batches = utils.load_dataset(original_ir_path, original_vi_path, batch_size)
        DF_model.train()
        count = 0
        batches = int(len(image_set_ir) // batch_size) #主要是不确定他是否必须，先留一下
        for batch in range(batches):
            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]#列表里取8张图
            image_paths_vi = image_set_vi[batch * batch_size:(batch * batch_size + batch_size)]
            ir = utils.get_train_images_auto(image_paths_ir, height=args1.height, width=args1.width)
            vi = utils.get_train_images_auto(image_paths_vi, height=args1.height, width=args1.width)
            ir1 = utils.get_train_images_auto1(image_paths_ir, height=args1.height, width=args1.width)
            vi1 = utils.get_train_images_auto1(image_paths_vi, height=args1.height, width=args1.width)
            count += 1
            optimizer.zero_grad()
            ir = Variable(ir, requires_grad=False)
            vi = Variable(vi, requires_grad=False)
            ir1 = Variable(ir1, requires_grad=False)
            vi1 = Variable(vi1, requires_grad=False)
            if args1.cuda:
                ir = ir.cuda()
                vi = vi.cuda()
                ir1 = ir1.cuda()
                vi1 = vi1.cuda()
            # get fusion image

            ir1, ir2, ir3, ir4, ir5, vi1, vi2, vi3, vi4, vi5 = DF_model.encoder(ir1, vi1)

            fusion1 = DF_model.fusion1(ir1, vi1)
            fusion2 = DF_model.fusion1(ir2, vi2)
            fusion3 = DF_model.fusion1(ir3, vi3)
            fusion4 = DF_model.fusion1(ir4, vi4)
            fusion5 = DF_model.fusion1(ir5, vi5)

            outputs = DF_model.decoder(fusion1, fusion2, fusion3, fusion4, fusion5)

            img_ir = Variable(ir.data.clone(), requires_grad=False)
            img_vi = Variable(vi.data.clone(), requires_grad=False)

            SSIM_loss_value = 0.
            Texture_loss_value = 0.
            Intensity_loss_value = 0.
            all_Texture_loss =0.
            all_SSIM_loss = 0.
            all_intensity_loss = 0.
            all_total_loss = 0.

            total_loss,SSIM_loss,Texture_loss,Intensity_loss = g_content_criterion(img_ir,img_vi,outputs)

            all_SSIM_loss += SSIM_loss.item()
            all_Texture_loss += Texture_loss.item()
            all_intensity_loss += Intensity_loss.item()
            all_total_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()

            if (batch + 1) % args1.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\t SSIM LOSS: {:.6f}\t Texture LOSS: {:.6f}\t Intensity LOSS: {:.6f}\t total: {:.6f}".format(
                    time.ctime(), e + 1, count, batches,
                                all_SSIM_loss / args1.log_interval,
                                all_Texture_loss / args1.log_interval,
                                all_intensity_loss / args1.log_interval,
                                all_total_loss / args1.log_interval
                )
                tbar.set_description(mesg)
                Loss_SSIM.append(all_SSIM_loss / args1.log_interval)
                Loss_Texture.append(all_Texture_loss / args1.log_interval)
                Loss_Indensity.append(all_intensity_loss / args1.log_interval)
                Loss_all.append(all_total_loss / args1.log_interval)
                count_loss = count_loss + 1

        if (e+1) % args1.log_interval == 0:
            # save model
            DF_model.eval()
            DF_model.cuda()
            STfuse_model_filename = "fuse_Epoch_" + str(e) + ".model"
            STfuse_model_path = os.path.join(args1.save_model_dir, STfuse_model_filename)
            torch.save(DF_model.state_dict(), STfuse_model_path)

    # SSIM loss
    loss_data_SSIM = Loss_SSIM
    loss_filename_path = 'final_SSIM.mat'
    scio.savemat(loss_filename_path, {'final_loss_SSIM': loss_data_SSIM})

    # Indensity loss
    loss_data_Indensity = Loss_Indensity
    loss_filename_path = "final_Indensity.mat"
    scio.savemat(loss_filename_path, {'final_loss_Indensity': loss_data_Indensity})

    # Indensity loss
    loss_data_Texture = Loss_Texture
    loss_filename_path = "final_Texture.mat"
    scio.savemat(loss_filename_path, {'final_loss_Texture': loss_data_Texture})

    loss_data = Loss_all
    loss_filename_path = "final_all.mat"
    scio.savemat(loss_filename_path, {'final_loss_all': loss_data})

    # save model
    DF_model.eval()
    DF_model.cpu()
    save_model_filename = "final_epoch.model"
    torch.save(DF_model.state_dict(), save_model_filename)

    print("\nDone, trained model saved at", save_model_filename)


if __name__ == "__main__":
    main()