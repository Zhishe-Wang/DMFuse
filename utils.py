import os
import torch
import torch.nn as nn
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from PIL import Image
import data.util as Util
from torch.autograd import Variable
from args_fusion import args1
# from scipy.misc import imread, imsave, imresize
from imageio import imread,imsave
import matplotlib as mpl
from torchvision import transforms
import imageio
import cv2
import torchvision



def list_images(dimgectory):
    images = []
    names = []
    dimg = listdir(dimgectory)
    dimg.sort()
    for file in dimg:
        name = file.lower()
        images.append(join(dimgectory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def make_floor(path1, path2):
    path = os.path.join(path1, path2)
    if os.path.exists(path) is False:
        os.makedirs(path)
    return path


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 1).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        # img = tensor.clone().clamp(0, 1).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U, D, V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
# def load_dataset(ir_imgs_path, vi_imgs_path, BATCH_SIZE, num_imgs=None):
#     if num_imgs is None:
#         num_imgs = len(ir_imgs_path)
#     ir_imgs_path = ir_imgs_path[:num_imgs]
#     vi_imgs_path = vi_imgs_path[:num_imgs]
#     # random
#     random.shuffle(ir_imgs_path)
#     random.shuffle(vi_imgs_path)
#     mod = num_imgs % BATCH_SIZE
#     print('BATCH SIZE %d.' % BATCH_SIZE)
#     print('Train images number %d.' % num_imgs)
#     print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))
#
#     if mod > 0:
#         print('Train set has been trimmed %d samples...\n' % mod)
#         ir_imgs_path = ir_imgs_path[:-mod]
#         vi_imgs_path = vi_imgs_path[:-mod]
#     batches = int(len(ir_imgs_path) // BATCH_SIZE)
#     return ir_imgs_path, vi_imgs_path, batches

def load_dataset(original_ir_path, original_vi_path, BATCH_SIZE, num_imgs=None):
    ir_path = list_images(original_ir_path)
    if num_imgs is None:
        num_imgs = len(ir_path)
    ir_path = ir_path[:num_imgs]
    # random
    random.shuffle(ir_path)
    vi_path = []
    for i in range(len(ir_path)):
        ir = ir_path[i]
        vis = ir.replace('ir', 'vi')
        vi_path.append(vis)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_ir_path = original_ir_path[:-mod]
    batches = int(len(original_ir_path) // BATCH_SIZE)
    return ir_path, vi_path, batches


def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args1.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
        # img_fusion = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    print(img_fusion.shape)
    # img_fusion = img_fusion.transpose(0, 2, 1).astype('uint8')
    # cv2.imwrite(output_path, img_fusion)
    # if img_fusion.shape[2] == 1:
    # img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    # 	img_fusion = imresize(img_fusion, [h, w])
    imsave(output_path, img_fusion)

#256get_image
def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = cv2.imread(path, 0)
    elif mode == 'RGB':
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image


def get_train_images_auto(paths, height=256, width=256, mode='L'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = images / 255
    return images

def get_train_images_auto1(paths, height=256, width=256, mode='L'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        # image = Image.open(path).convert("RGB")
        image = torch.tensor(image)
        image0 = torch.cat([image,image],0)
        image = torch.cat([image0, image], 0)
        image = Util.transform_augment_cd1(image, split='train', min_max=(-1, 1))
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    # images = images / 255
    return images

# def get_test_images(paths, height=None, width=None, mode='L'):
#     global image
#     ImageToTensor = transforms.Compose([transforms.ToTensor()])
#     if isinstance(paths, str):
#         paths = [paths]
#     images = []
#     for path in paths:
#         image = get_image(path, height, width, mode=mode)
#         w, h = image.shape[0], image.shape[1]
#         # w_s = 256 - w % 256
#         # h_s = 256 - h % 256
#         if w % 256 != 0:
#             w_s = 256 - w % 256
#         else:
#             w_s = 0
#         if h % 256 != 0:
#             h_s = 256 - h % 256
#         else:
#             h_s = 0
#         image = cv2.copyMakeBorder(image, 0, w_s, 0, h_s, cv2.BORDER_CONSTANT,value=256)
#         if mode == 'L':
#             image = np.reshape(image, [1, image.shape[0], image.shape[1]])
#         else:
#             image = ImageToTensor(image).float().numpy()*255
#     images.append(image)
#     images = np.stack(images, axis=0)
#     images = torch.from_numpy(images).float()
#     images = images/ 255
#     return images
def get_test_images(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        # image = Image.open(path).convert("RGB")
        # image1 = get_image(path, height, width, mode=mode)
        # w, h = image1.shape[0], image1.shape[1]
        # w_s = 256 - w % 256
        # h_s = 256 - h % 256
        # image = np.array(image)
        # image = cv2.copyMakeBorder(image, 0, w_s, 0, h_s, cv2.BORDER_CONSTANT,
        #                              value=256)
        # image = Util.transform_augment_cd(image, split='train', min_max=(-1, 1))

        image = get_image(path, height, width, mode=mode)
        w, h = image.shape[0], image.shape[1]
        if w%32 != 0:
            w_s = 32 - w % 32
        else:
            w_s = 0
        if h % 32 != 0:
            h_s = 32 - h % 32
        else:
            h_s = 0


        # w_s = 256- w%256
        # h_s = 256- h%256
        image = cv2.copyMakeBorder(image, 0, w_s, 0, h_s, cv2.BORDER_CONSTANT,value=256)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])


        else:
            image = ImageToTensor(image).float().numpy()*255

        image = Util.transform_augment_cd1(image, split='train', min_max=(-1, 1))


        image = torch.tensor(image)
        image0 = torch.cat([image, image], 0)
        image = torch.cat([image0, image], 0)



    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    # images = images/ 255
    return images

# def get_test_images(paths, height=None, width=None, flag=False):
#     global h, w, c
#     if isinstance(paths, str):
#         paths = [paths]
#     images = []
#     for path in paths:
#         image = get_image(path, height, width, flag)
#         if height is not None and width is not None:
#             image = imresize(image, [height, width], interp='nearest')
#         base_size = 256
#         h = image.shape[0]
#         w = image.shape[1]
#         if 1 * base_size < h < 2 * base_size and 1 * base_size < w < 2 * base_size:
#             c = 4
#             images = get_img_parts1(image, h, w)
#         if 2 * base_size < h < 3 * base_size and 2 * base_size < w < 3 * base_size:
#             c = 9
#             images = get_img_parts2(image, h, w)
#         if 2 * base_size < h < 3 * base_size and 3 * base_size < w < 4 * base_size:
#             c = 12
#             images = get_img_parts3(image, h, w)
#         if 1 * base_size < h < 2 * base_size and 2 * base_size < w < 3 * base_size:
#             c = 6
#             images = get_img_parts4(image, h, w)
#         if 3 * base_size < h < 4 * base_size and 4 * base_size < w < 5 * base_size:
#             c = 20
#             images = get_img_parts5(image, h, w)
#         if 0 * base_size < h < 1 * base_size and 1 * base_size < w < 2 * base_size:
#             c = 2
#             images = get_img_parts6(image, h, w)
#         if 0 * base_size < h < 1 * base_size and 2 * base_size < w < 3 * base_size:
#             c = 3
#             images = get_img_parts7(image, h, w)
#         if h == 1 * base_size and 2 * base_size < w < 3 * base_size:
#             c = 3
#             images = get_img_parts8(image, h, w)
#         if 2 * base_size < h < 3 * base_size and w == 3 * base_size:
#             c = 9
#             images = get_img_parts9(image, h, w)
#         if 1 * base_size < h < 2 * base_size and w == 2 * base_size:
#             c = 4
#             images = get_img_parts10(image, h, w)
#         if h == 1 * base_size and 1 * base_size < w < 2 * base_size:
#             c = 1
#             images = get_img_parts12(image, h, w)
#
#         if h == 1 * base_size and w == 1 * base_size :
#             c = 2
#             images = get_img_parts11(image, h, w)
#
#         if h == 3 * base_size and w == 4 * base_size :
#             c = 12
#             images = get_img_parts13(image, h, w)
#
#         # if 0 * base_size < h < 1 * base_size and 0 * base_size < w < 1 * base_size:
#         #     c = 1
#         #     images = get_img_parts1(image, h, w)
#         # if 0 * base_size < h < 1 * base_size and 1 * base_size < w < 2 * base_size:
#         #     c = 2
#         #     images = get_img_parts2(image, h, w)
#         # if 1 * base_size < h < 2 * base_size and 1 * base_size < w < 2 * base_size:
#         #     c = 4
#         #     images = get_img_parts3(image, h, w)
#         # if 1 * base_size < h < 2 * base_size and 2 * base_size < w < 3 * base_size:
#         #     c = 6
#         #     images = get_img_parts4(image, h, w)
#
#     return images, h, w, c

def save_feat(index,C,ir_atten_feat,vi_atten_feat,result_path):
    ir_atten_feat = ir_atten_feat * 255
    vi_atten_feat = vi_atten_feat * 255

    ir_feat_path = make_floor(result_path, "ir_feat")
    index_irfeat_path = make_floor(ir_feat_path, str(index))

    vi_feat_path = make_floor(result_path, "vi_feat")
    index_vifeat_path = make_floor(vi_feat_path, str(index))

    for c in range(C):
        ir_temp = ir_atten_feat[:, c, :, :].squeeze()
        vi_temp = vi_atten_feat[:, c, :, :].squeeze()

        feat_ir = ir_temp.cpu().clamp(0, 255).data.numpy()
        feat_vi = vi_temp.cpu().clamp(0, 255).data.numpy()

        feat_ir = feat_ir.astype(np.uint8)
        feat_vi = feat_vi.astype(np.uint8)

        ir_feat_filenames = 'ir_feat_C' + str(c) + '.png'
        ir_atten_path = index_irfeat_path + '/' + ir_feat_filenames
        imsave(ir_atten_path, feat_ir)
        # cv2.imwrite(ir_atten_path,cv2.applyColorMap(cv2.resize(feat_ir, (256, 256), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_JET))

        vi_feat_filenames = 'vi_feat_C' + str(c) + '.png'
        vi_atten_path = index_vifeat_path + '/' + vi_feat_filenames
        imsave(vi_atten_path, feat_vi)
        # cv2.imwrite(vi_atten_path,cv2.applyColorMap(cv2.resize(feat_vi, (256, 256), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_JET))

def get_img_parts1(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, base_size * 2 - w, 0, base_size * 2 - h), value=0)
    image = torch.tensor(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1,img1.shape[0], img1.shape[1]])
    img1_0 = torch.cat([img1,img1],1)
    img1 = torch.cat([img1_0, img1], 1)
    img1 = Util.transform_augment_cd1(img1, split='train', min_max=(-1, 1))
    img2 = image[0:base_size, base_size: base_size * 2]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img2_0 = torch.cat([img2, img2], 1)
    img2 = torch.cat([img2_0, img2], 1)
    img2 = Image.open(img2).convert("RGB")
    img2 = Util.transform_augment_cd1(img2, split='train', min_max=(-1, 1))
    img3 = image[base_size:base_size * 2, 0: base_size]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img3_0 = torch.cat([img3, img3], 1)
    img3 = torch.cat([img3_0, img3], 1)
    img3 = Util.transform_augment_cd1(img3, split='train', min_max=(-1, 1))
    img4 = image[base_size:base_size * 2, base_size: base_size * 2]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img4_0 = torch.cat([img4, img4], 1)
    img4 = torch.cat([img4_0, img4], 1)
    img4 = Util.transform_augment_cd1(img4, split='train', min_max=(-1, 1))
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    return images


def get_img_parts2(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, base_size * 3 - w, 0, base_size * 3 - h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:base_size, base_size: base_size * 2]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:base_size, base_size * 2: base_size * 3]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[base_size:base_size * 2, 0: base_size]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[base_size:base_size * 2, base_size: base_size * 2]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[base_size:base_size * 2, base_size * 2: base_size * 3]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[base_size * 2:base_size * 3, 0: base_size]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[base_size * 2:base_size * 3, base_size: base_size * 2]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[base_size * 2:base_size * 3, base_size * 2: base_size * 3]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    return images


def get_img_parts3(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, base_size * 4 - w, 0, base_size * 3 - h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:base_size, base_size: base_size * 2]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:base_size, base_size * 2: base_size * 3]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[0:base_size, base_size * 3: base_size * 4]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[base_size:base_size * 2, 0: base_size]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[base_size:base_size * 2, base_size: base_size * 2]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[base_size:base_size * 2, base_size * 2: base_size * 3]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[base_size:base_size * 2, base_size * 3: base_size * 4]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[base_size * 2:base_size * 3, 0: base_size]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    img10 = image[base_size * 2:base_size * 3, base_size: base_size * 2]
    img10 = torch.reshape(img10, [1, 1, img10.shape[0], img10.shape[1]])
    img11 = image[base_size * 2:base_size * 3, base_size * 2: base_size * 3]
    img11 = torch.reshape(img11, [1, 1, img11.shape[0], img11.shape[1]])
    img12 = image[base_size * 2:base_size * 3, base_size * 3: base_size * 4]
    img12 = torch.reshape(img12, [1, 1, img12.shape[0], img12.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    images.append(img10.float())
    images.append(img11.float())
    images.append(img12.float())
    return images


def get_img_parts4(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, base_size * 3 - w, 0, base_size * 2 - h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:base_size, base_size: base_size * 2]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:base_size, base_size * 2: base_size * 3]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[base_size:base_size * 2, 0: base_size]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[base_size:base_size * 2, base_size: base_size * 2]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[base_size:base_size * 2, base_size * 2: base_size * 3]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    return images


def get_img_parts5(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, base_size * 5 - w, 0, base_size * 4 - h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:base_size, base_size: base_size * 2]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:base_size, base_size * 2: base_size * 3]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[0:base_size, base_size * 3: base_size * 4]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[0:base_size, base_size * 4: base_size * 5]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[base_size:base_size * 2, 0: base_size]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[base_size:base_size * 2, base_size: base_size * 2]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[base_size:base_size * 2, base_size * 2: base_size * 3]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[base_size:base_size * 2, base_size * 3: base_size * 4]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    img10 = image[base_size:base_size * 2, base_size * 4: base_size * 5]
    img10 = torch.reshape(img10, [1, 1, img10.shape[0], img10.shape[1]])
    img11 = image[base_size * 2:base_size * 3, 0: base_size]
    img11 = torch.reshape(img11, [1, 1, img11.shape[0], img11.shape[1]])
    img12 = image[base_size * 2:base_size * 3, base_size: base_size * 2]
    img12 = torch.reshape(img12, [1, 1, img12.shape[0], img12.shape[1]])
    img13 = image[base_size * 2:base_size * 3, base_size * 2: base_size * 3]
    img13 = torch.reshape(img13, [1, 1, img13.shape[0], img13.shape[1]])
    img14 = image[base_size * 2:base_size * 3, base_size * 3: base_size * 4]
    img14 = torch.reshape(img14, [1, 1, img14.shape[0], img14.shape[1]])
    img15 = image[base_size * 2:base_size * 3, base_size * 4: base_size * 5]
    img15 = torch.reshape(img15, [1, 1, img15.shape[0], img15.shape[1]])
    img16 = image[base_size * 3:base_size * 4, 0: base_size]
    img16 = torch.reshape(img16, [1, 1, img16.shape[0], img16.shape[1]])
    img17 = image[base_size * 3:base_size * 4, base_size: base_size * 2]
    img17 = torch.reshape(img17, [1, 1, img17.shape[0], img17.shape[1]])
    img18 = image[base_size * 3:base_size * 4, base_size * 2: base_size * 3]
    img18 = torch.reshape(img18, [1, 1, img18.shape[0], img18.shape[1]])
    img19 = image[base_size * 3:base_size * 4, base_size * 3: base_size * 4]
    img19 = torch.reshape(img19, [1, 1, img19.shape[0], img19.shape[1]])
    img20 = image[base_size * 3:base_size * 4, base_size * 4: base_size * 5]
    img20 = torch.reshape(img20, [1, 1, img20.shape[0], img20.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    images.append(img10.float())
    images.append(img11.float())
    images.append(img12.float())
    images.append(img13.float())
    images.append(img14.float())
    images.append(img15.float())
    images.append(img16.float())
    images.append(img17.float())
    images.append(img18.float())
    images.append(img19.float())
    images.append(img20.float())
    return images


def get_img_parts6(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, base_size * 2 - w, 0, base_size - h), value=0)
    image = torch.tensor(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:base_size, base_size: base_size * 2]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    return images


def get_img_parts7(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, base_size * 3-w, 0, base_size-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:base_size, base_size: base_size * 2]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:base_size, base_size * 2: base_size * 3]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    return images


def get_img_parts8(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, base_size * 3-w, 0, base_size), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:base_size, base_size: base_size * 2]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:base_size, base_size * 2: base_size * 3]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    return images


def get_img_parts9(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, w, 0, base_size * 3 - h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:base_size, base_size: base_size * 2]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:base_size, base_size * 2: base_size * 3]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[base_size:base_size * 2, 0: base_size]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[base_size:base_size * 2, base_size: base_size * 2]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[base_size:base_size * 2, base_size * 2: base_size * 3]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[base_size * 2:base_size * 3, 0: base_size]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[base_size * 2:base_size * 3, base_size: base_size * 2]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[base_size * 2:base_size * 3, base_size * 2: base_size * 3]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    return images

def get_img_parts10(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, w, 0, base_size * 2 - h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:base_size, base_size: base_size * 2]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[base_size:base_size * 2, 0: base_size]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[base_size:base_size * 2, base_size: base_size * 2]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    return images


def get_img_parts11(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, base_size * 2 - w, 0, h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:base_size, base_size: base_size * 2]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    images.append(img1.float())
    images.append(img2.float())

    return images

def get_img_parts12(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, 0, 0, 0), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    images.append(img1.float())

    return images

def get_img_parts13(image, h, w):
    base_size = 256
    pad = nn.ConstantPad2d(padding=(0, 0, 0, 0), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:base_size, 0: base_size]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:base_size, base_size: base_size * 2]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:base_size, base_size * 2: base_size * 3]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[0:base_size, base_size * 3: base_size * 4]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[base_size:base_size * 2, 0: base_size]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[base_size:base_size * 2, base_size: base_size * 2]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[base_size:base_size * 2, base_size * 2: base_size * 3]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[base_size:base_size * 2, base_size * 3: base_size * 4]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[base_size * 2:base_size * 3, 0: base_size]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    img10 = image[base_size * 2:base_size * 3, base_size: base_size * 2]
    img10 = torch.reshape(img10, [1, 1, img10.shape[0], img10.shape[1]])
    img11 = image[base_size * 2:base_size * 3, base_size * 2: base_size * 3]
    img11 = torch.reshape(img11, [1, 1, img11.shape[0], img11.shape[1]])
    img12 = image[base_size * 2:base_size * 3, base_size * 3: base_size * 4]
    img12 = torch.reshape(img12, [1, 1, img12.shape[0], img12.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    images.append(img10.float())
    images.append(img11.float())
    images.append(img12.float())
    return images

def recons_fusion_images1(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:base_size, 0: base_size] += img1
        img_f[:, 0:base_size, base_size: w] += img2[:, 0:base_size, 0:w - base_size]
        img_f[:, base_size:h, 0: base_size] += img3[:, 0:h - base_size, 0:base_size]
        img_f[:, base_size:h, base_size: w] += img4[:, 0:h - base_size, 0:w - base_size]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images2(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:base_size, 0: base_size] += img1
        img_f[:, 0:base_size, base_size: base_size * 2] += img2
        img_f[:, 0:base_size, base_size * 2: w] += img3[:, 0:base_size, 0:w - base_size * 2]
        img_f[:, base_size:base_size * 2, 0: base_size] += img4
        img_f[:, base_size:base_size * 2, base_size: base_size * 2] += img5
        img_f[:, base_size:base_size * 2, base_size * 2: w] += img6[:, 0:base_size, 0:w - base_size * 2]
        img_f[:, base_size * 2:h, 0: base_size] += img7[:, 0:h - base_size * 2, 0:base_size]
        img_f[:, base_size * 2:h, base_size: base_size * 2] += img8[:, 0:h - base_size * 2, 0:base_size]
        img_f[:, base_size * 2:h, base_size * 2: w] += img9[:, 0:h - base_size * 2, 0:w - base_size * 2]
        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images3(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img10 = img_lists[9][i]
        img11 = img_lists[10][i]
        img12 = img_lists[11][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:base_size, 0: base_size] += img1
        img_f[:, 0:base_size, base_size: base_size * 2] += img2
        img_f[:, 0:base_size, base_size * 2: base_size * 3] += img3
        img_f[:, 0:base_size, base_size * 3: w] += img4[:, 0:base_size, 0:w - base_size * 3]
        img_f[:, base_size:base_size * 2, 0: base_size] += img5
        img_f[:, base_size:base_size * 2, base_size: base_size * 2] += img6
        img_f[:, base_size:base_size * 2, base_size * 2: base_size * 3] += img7
        img_f[:, base_size:base_size * 2, base_size * 3: w] += img8[:, 0:base_size, 0:w - base_size * 3]
        img_f[:, base_size * 2:h, 0: base_size] += img9[:, 0:h - base_size * 2, 0:base_size]
        img_f[:, base_size * 2:h, base_size: base_size * 2] += img10[:, 0:h - base_size * 2, 0:base_size]
        img_f[:, base_size * 2:h, base_size * 2: base_size * 3] += img11[:, 0:h - base_size * 2, 0:base_size]
        img_f[:, base_size * 2:h, base_size * 3: w] += img12[:, 0:h - base_size * 2, 0:w - base_size * 3]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images4(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]

        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:base_size, 0: base_size] += img1
        img_f[:, 0:base_size, base_size: base_size * 2] += img2
        img_f[:, 0:base_size, base_size * 2: w] += img3[:, 0:base_size, 0:w - base_size * 2]
        img_f[:, base_size:h, 0: base_size] += img4[:, 0:h - base_size, 0:base_size]
        img_f[:, base_size:h, base_size: base_size * 2] += img5[:, 0:h - base_size, 0:base_size]
        img_f[:, base_size:h, base_size * 2: w] += img6[:, 0:h - base_size, 0:w - base_size * 2]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images5(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img10 = img_lists[9][i]
        img11 = img_lists[10][i]
        img12 = img_lists[11][i]
        img13 = img_lists[12][i]
        img14 = img_lists[13][i]
        img15 = img_lists[14][i]
        img16 = img_lists[15][i]
        img17 = img_lists[16][i]
        img18 = img_lists[17][i]
        img19 = img_lists[18][i]
        img20 = img_lists[19][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:base_size, 0: base_size] += img1
        img_f[:, 0:base_size, base_size: base_size * 2] += img2
        img_f[:, 0:base_size, base_size * 2: base_size * 3] += img3
        img_f[:, 0:base_size, base_size * 3: base_size * 4] += img4
        img_f[:, 0:base_size, base_size * 4: w] += img5[:, 0:base_size, 0:w - base_size * 4]
        img_f[:, base_size:base_size * 2, 0: base_size] += img6
        img_f[:, base_size:base_size * 2, base_size: base_size * 2] += img7
        img_f[:, base_size:base_size * 2, base_size * 2: base_size * 3] += img8
        img_f[:, base_size:base_size * 2, base_size * 3: base_size * 4] += img9
        img_f[:, base_size:base_size * 2, base_size * 4: w] += img10[:, 0:base_size, 0:w - base_size * 4]
        img_f[:, base_size * 2:base_size * 3, 0: base_size] += img11
        img_f[:, base_size * 2:base_size * 3, base_size: base_size * 2] += img12
        img_f[:, base_size * 2:base_size * 3, base_size * 2: base_size * 3] += img13
        img_f[:, base_size * 2:base_size * 3, base_size * 3: base_size * 4] += img14
        img_f[:, base_size * 2:base_size * 3, base_size * 4: w] += img15[:, 0:base_size, 0:w - base_size * 4]
        img_f[:, base_size * 3:h, 0: base_size] += img16[:, 0:h - base_size * 3, 0:base_size]
        img_f[:, base_size * 3:h, base_size: base_size * 2] += img17[:, 0:h - base_size * 3, 0:base_size]
        img_f[:, base_size * 3:h, base_size * 2: base_size * 3] += img18[:, 0:h - base_size * 3, 0:base_size]
        img_f[:, base_size * 3:h, base_size * 3: base_size * 4] += img19[:, 0:h - base_size * 3, 0:base_size]
        img_f[:, base_size * 3:h, base_size * 4: w] += img20[:, 0:h - base_size * 3, 0:w - base_size * 4]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images6(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:h, 0: base_size] += img1[:, 0:h, 0:base_size]
        img_f[:, 0:h, base_size: w] += img2[:, 0:h, 0:w-base_size]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images7(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:h, 0: base_size] += img1[:, 0:h, 0:base_size]
        img_f[:, 0:h, base_size: base_size * 2] += img2[:, 0:h, 0:base_size]
        img_f[:, 0:h, base_size * 2: w] += img3[:, 0:h, 0:w - base_size * 2]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images8(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:h, 0: base_size] += img1[:, 0:h, 0:base_size]
        img_f[:, 0:h, base_size: base_size * 2] += img2[:, 0:h, 0:base_size]
        img_f[:, 0:h, base_size * 2: w] += img3[:, 0:h, 0:w - base_size * 2]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images9(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:base_size, 0: base_size] += img1
        img_f[:, 0:base_size, base_size: base_size * 2] += img2
        img_f[:, 0:base_size, base_size * 2: w] += img3
        img_f[:, base_size:base_size * 2, 0: base_size] += img4
        img_f[:, base_size:base_size * 2, base_size: base_size * 2] += img5
        img_f[:, base_size:base_size * 2, base_size * 2: w] += img6
        img_f[:, base_size * 2:h, 0: base_size] += img7[:, 0:h - base_size * 2, 0:base_size]
        img_f[:, base_size * 2:h, base_size: base_size * 2] += img8[:, 0:h - base_size * 2, 0:base_size]
        img_f[:, base_size * 2:h, base_size * 2: w] += img9[:, 0:h - base_size * 2, 0:base_size]
        img_f_list.append(img_f)
    return img_f_list

def recons_fusion_images10(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:base_size, 0: base_size] += img1
        img_f[:, 0:base_size, base_size: base_size * 2] += img2
        img_f[:, base_size:h, 0: base_size] += img3[:, 0:h - base_size, 0:base_size]
        img_f[:, base_size:h, base_size: w] += img4[:, 0:h - base_size, 0:base_size]
        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images11(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]

        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:base_size, 0: base_size] += img1
        img_f[:, 0:base_size, base_size: w] += img2[:, 0:base_size, 0:w - base_size]
        img_f_list.append(img_f)
    return img_f_list



def recons_fusion_images12(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img10 = img_lists[9][i]
        img11 = img_lists[10][i]
        img12 = img_lists[11][i]

        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:base_size, 0: base_size] += img1
        img_f[:, 0:base_size, base_size: base_size * 2] += img2
        img_f[:, 0:base_size, base_size * 2: base_size * 3] += img3
        img_f[:, 0:base_size, base_size * 3: base_size * 4] += img4
        img_f[:, base_size:base_size * 2, 0: base_size] += img5
        img_f[:, base_size:base_size * 2, base_size: base_size * 2] += img6
        img_f[:, base_size:base_size * 2, base_size * 2: base_size * 3] += img7
        img_f[:, base_size:base_size * 2, base_size * 3: base_size * 4] += img8
        img_f[:, base_size * 2:base_size * 3, 0: base_size] += img9
        img_f[:, base_size * 2:base_size * 3, base_size: base_size * 2] += img10
        img_f[:, base_size * 2:base_size * 3, base_size * 2: base_size * 3] += img11
        img_f[:, base_size * 2:base_size * 3, base_size * 3: base_size * 4] += img12
        img_f_list.append(img_f)
    return img_f_list

def recons_fusion_images13(img_lists, h, w):
    base_size = 256
    img_f_list = []
    for i in range(len(img_lists[0])):
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img10 = img_lists[9][i]
        img11 = img_lists[10][i]
        img12 = img_lists[11][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:base_size, 0: base_size] += img1
        img_f[:, 0:base_size, base_size: base_size * 2] += img2
        img_f[:, 0:base_size, base_size * 2: base_size * 3] += img3
        img_f[:, 0:base_size, base_size * 3: w] += img4[:, 0:base_size, 0:w - base_size * 3]
        img_f[:, base_size:base_size * 2, 0: base_size] += img5
        img_f[:, base_size:base_size * 2, base_size: base_size * 2] += img6
        img_f[:, base_size:base_size * 2, base_size * 2: base_size * 3] += img7
        img_f[:, base_size:base_size * 2, base_size * 3: w] += img8[:, 0:base_size, 0:w - base_size * 3]
        img_f[:, base_size * 2:h, 0: base_size] += img9[:, 0:h - base_size * 2, 0:base_size]
        img_f[:, base_size * 2:h, base_size: base_size * 2] += img10[:, 0:h - base_size * 2, 0:base_size]
        img_f[:, base_size * 2:h, base_size * 2: base_size * 3] += img11[:, 0:h - base_size * 2, 0:base_size]
        img_f[:, base_size * 2:h, base_size * 3: w] += img12[:, 0:h - base_size * 2, 0:w - base_size * 3]

        img_f_list.append(img_f)
    return img_f_list


# def get_img_parts1(image, h, w):
#     pad = nn.ConstantPad2d(padding=(0, 448 - w, 0, 448 - h), value=0)
#     image = torch.from_numpy(image)
#     image = pad(image)
#     images = []
#     img1 = image[0:448, 0: 448]
#     img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
#     images.append(img1.float())
#     return images
#
#
# def get_img_parts2(image, h, w):
#     pad = nn.ConstantPad2d(padding=(0, 896 - w, 0, 448 - h), value=0)
#     image = torch.from_numpy(image)
#     image = pad(image)
#     images = []
#     img1 = image[0:448, 0: 448]
#     img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
#     img2 = image[0:448, 448: 896]
#     img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
#     images.append(img1.float())
#     images.append(img2.float())
#     return images
#
#
# def get_img_parts3(image, h, w):
#     pad = nn.ConstantPad2d(padding=(0, 896 - w, 0, 896 - h), value=0)
#     image = torch.from_numpy(image)
#     image = pad(image)
#     images = []
#     img1 = image[0:448, 0: 448]
#     img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
#     img2 = image[0:448, 448: 896]
#     img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
#     img3 = image[448:896, 0: 448]
#     img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
#     img4 = image[448:896, 448: 896]
#     img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
#     images.append(img1.float())
#     images.append(img2.float())
#     images.append(img3.float())
#     images.append(img4.float())
#     return images
#
#
# def get_img_parts4(image, h, w):
#     pad = nn.ConstantPad2d(padding=(0, 1344 - w, 0, 896 - h), value=0)
#     image = torch.from_numpy(image)
#     image = pad(image)
#     images = []
#     img1 = image[0:448, 0: 448]
#     img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
#     img2 = image[0:448, 448: 896]
#     img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
#     img3 = image[0:448, 896: 1344]
#     img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
#     img4 = image[448:896, 0: 448]
#     img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
#     img5 = image[448:896, 448: 896]
#     img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
#     img6 = image[448:896, 896: 1344]
#     img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
#     images.append(img1.float())
#     images.append(img2.float())
#     images.append(img3.float())
#     images.append(img4.float())
#     images.append(img5.float())
#     images.append(img6.float())
#     return images
#
#
# def recons_fusion_images1(img_lists, h, w):
#     img_f_list = []
#     for i in range(len(img_lists[0])):
#         img1 = img_lists[0][i]
#
#         img_f = torch.zeros(1, h, w).cuda()
#         print(img_f.size())
#
#         img_f[:, 0:h, 0: w] += img1[:, 0:h, 0:w]
#
#         img_f_list.append(img_f)
#     return img_f_list
#
#
# def recons_fusion_images2(img_lists, h, w):
#     img_f_list = []
#     for i in range(len(img_lists[0])):
#         img1 = img_lists[0][i]
#         img2 = img_lists[1][i]
#         img_f = torch.zeros(1, h, w).cuda()
#
#         img_f[:, 0:h, 0: 448] += img1[:, 0:h, 0:448]
#         img_f[:, 0:h, 448: w] += img2[:, 0:h, 0:w - 448]
#         img_f_list.append(img_f)
#     return img_f_list
#
#
# def recons_fusion_images3(img_lists, h, w):
#     img_f_list = []
#     for i in range(len(img_lists[0])):
#         img1 = img_lists[0][i]
#         img2 = img_lists[1][i]
#         img3 = img_lists[2][i]
#         img4 = img_lists[3][i]
#         img_f = torch.zeros(1, h, w).cuda()
#
#         img_f[:, 0:448, 0: 448] += img1
#         img_f[:, 0:448, 448: w] += img2[:, 0:448, 0:w - 448]
#         img_f[:, 448:h, 0: 448] += img3[:, 0:h - 448, 0:448]
#         img_f[:, 448:h, 448: w] += img4[:, 0:h - 448, 0:w - 448]
#
#         img_f_list.append(img_f)
#     return img_f_list
#
#
# def recons_fusion_images4(img_lists, h, w):
#     img_f_list = []
#     for i in range(len(img_lists[0])):
#         img1 = img_lists[0][i]
#         img2 = img_lists[1][i]
#         img3 = img_lists[2][i]
#         img4 = img_lists[3][i]
#         img5 = img_lists[4][i]
#         img6 = img_lists[5][i]
#
#         img_f = torch.zeros(1, h, w).cuda()
#
#         img_f[:, 0:448, 0: 448] += img1
#         img_f[:, 0:448, 448: 896] += img2
#         img_f[:, 0:448, 896: w] += img3[:, 0:448, 0:w - 896]
#         img_f[:, 448:h, 0: 448] += img4[:, 0:h - 448, 0:448]
#         img_f[:, 448:h, 448: 896] += img5[:, 0:h - 448, 0:448]
#         img_f[:, 448:h, 896: w] += img6[:, 0:h - 448, 0:w - 896]
#
#         img_f_list.append(img_f)
#     return img_f_list


def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args1.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    # img_fusion = img_fusion * 127.5 + 127.5
    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    print(img_fusion.shape)
    img_fusion = img_fusion.reshape([1, img_fusion.shape[0], img_fusion.shape[1]])  # 有些出来的是二维，需要变成3维
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    imageio.imwrite(output_path, img_fusion)


'''# colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00', '#FF0000',
                                                                 '#8B0000'], 224)'''


def save_imgs(path, img_fusion):
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    imsave(path, img_fusion)


def save_image_scales(img_fusion, output_path):
    img_fusion = img_fusion.float()
    img_fusion = img_fusion.cpu().data[0].numpy()
    imageio.imwrite(output_path, img_fusion)

def save_images(path, data, out):
    w, h = out.shape[0], out.shape[1]
    if data.shape[1] == 1:
        data = data.reshape([data.shape[2], data.shape[3]])
    ori = data[0:w, 0:h]
    # ori = ori *255
    cv2.imwrite(path, ori)

def make_floor(path1,path2):
    path = os.path.join(path1,path2)
    if os.path.exists(path) is False:
        os.makedirs(path)
    return path