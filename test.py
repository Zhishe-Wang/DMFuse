import torch
import time
from Net import net
import os
import numpy as np
import utils
from torch.autograd import Variable





def load_model(path):


    fuse_net = net()
    fuse_net.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in fuse_net.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fuse_net._get_name(), para * type_size / 1000 / 1000))

    fuse_net.eval()
    fuse_net.cuda()

    return fuse_net

def _generate_fusion_image(model, vi, ir):
    ir1, ir2, ir3, ir4, ir5, vi1, vi2, vi3, vi4, vi5 = model.encoder(ir, vi)
    fusion1 = model.fusion(ir1, vi1)
    fusion2 = model.fusion(ir2, vi2)
    fusion3 = model.fusion(ir3, vi3)
    fusion4 = model.fusion(ir4, vi4)
    fusion5 = model.fusion(ir5, vi5)
    outputs = model.decoder(fusion1, fusion2, fusion3, fusion4, fusion5)
    return outputs


def run_demo(model, vi_path, ir_path, output_path_root, img_name):
    vi_img = utils.get_test_images(vi_path, height=None, width=None)
    ir_img = utils.get_test_images(ir_path, height=None, width=None)

    out = utils.get_image(vi_path, height=None, width=None)


    vi_img = vi_img.cuda()
    ir_img = ir_img.cuda()
    vi_img = Variable(vi_img, requires_grad=False)
    ir_img = Variable(ir_img, requires_grad=False)


    img_fusion = _generate_fusion_image(model, vi_img, ir_img)
    ############################ multi outputs ##############################################
    file_name = img_name
    output_path = output_path_root + file_name
    if torch.cuda.is_available():
        img = img_fusion.cpu().clamp(0, 255).numpy()
    else:
        img = img_fusion.clamp(0, 255).numpy()
    img = img * 255
    utils.save_images(output_path, img, out)
    print(output_path)

def main():


    vi_path = "./images/vi/"
    ir_path = "./images/ir/"


    output_path = './outputs/'

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # in_c = 1
    # out_c = in_c
    model_path = "./models/model.model"
    with torch.no_grad():

        model = load_model(model_path)



        for img_name in os.listdir(ir_path):
            visible_path = vi_path + img_name
            infrared_path = ir_path + img_name
            start = time.time()
            run_demo(model, visible_path, infrared_path, output_path, img_name)
            end = time.time()
            print(end - start)



    # end = time.time()

    print('Done......')


if __name__ == "__main__":
    main()