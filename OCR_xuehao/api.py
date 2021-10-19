# -*- encoding: utf-8 -*-
"""
@Author             :  Hao Shen 
@Last Modified by   :  Hao Shen
@Last Modified time :  2021/10/19 16:41:31
@Email              :  shenhao0223sh@gamil.com
@Describe           :  api code
"""

# here put the import lib

import os

import torch
import torchvision.transforms as transforms
from PIL import Image

from analysis_recognition_dataset import load_lbl2id_map, statistics_max_len_label
from ocr_by_transformer import *


def predict_student_num(img_path, id2lbl_map, model, max_len, start_symbol=1, end_symbol=2):
    # 定义随机颜色变换
    color_trans = transforms.ColorJitter(0.1, 0.1, 0.1)
    # 定义 Normalize
    trans_Normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert('RGB')

    # 对图片进行大致等比例的缩放
    # 将高缩放到32，宽大致等比例缩放，但要被32整除
    w, h = img.size
    max_ratio = 8
    ratio = round((w / h) * 3)  # 将宽拉长3倍，然后四舍五入
    if ratio == 0:
        ratio = 1
    if ratio > max_ratio:
        ratio = max_ratio
    h_new = 32
    w_new = h_new * ratio
    img_resize = img.resize((w_new, h_new), Image.BILINEAR)

    # 对图片右半边进行padding，使得宽/高比例固定=self.max_ratio
    img_padd = Image.new('RGB', (32 * max_ratio, 32), (0, 0, 0))
    img_padd.paste(img_resize, (0, 0))

    # 随机颜色变换
    img_input = color_trans(img_padd)
    # Normalize
    src = trans_Normalize(img_input).unsqueeze(0)

    # 构造encoder的mask
    src_mask = [1] * ratio + [0] * (max_ratio - ratio)
    src_mask = torch.tensor(src_mask)
    src_mask = (src_mask != 0).unsqueeze(0)

    pred_result = greedy_decode(
        model, src, src_mask, max_len, start_symbol, end_symbol)

    lbl = ""
    for id in pred_result:
        label = id2lbl_map[int(id)]
        lbl += label
    return lbl


if __name__ == "__main__":
    # TODO set parameters
    base_data_dir = './ICDAR_2015'  # 数据集根目录，请将数据下载到此位置
    device = torch.device('cpu')
    nrof_epochs = 500
    batch_size = 16
    model_save_path = './log/ex1_ocr_model.pth'

    # 读取label-id映射关系记录文件
    lbl2id_map_path = os.path.join(base_data_dir, 'lbl2id_map.txt')
    lbl2id_map, id2lbl_map = load_lbl2id_map(lbl2id_map_path)

    # 统计数据集中出现的所有的label中包含字符最多的有多少字符，数据集构造gt信息需要用到
    train_lbl_path = os.path.join(base_data_dir, 'train_gt.txt')
    valid_lbl_path = os.path.join(base_data_dir, 'valid_gt.txt')
    train_max_label_len = statistics_max_len_label(train_lbl_path)
    valid_max_label_len = statistics_max_len_label(valid_lbl_path)
    # 数据集中字符数最多的一个case作为制作的gt的sequence_len
    sequence_len = max(train_max_label_len, valid_max_label_len)

    # 构造 dataloader
    max_ratio = 8  # 图片预处理时 宽/高的最大值，不超过就保比例resize，超过会强行压缩

    # build model
    # use transformer as ocr recognize model
    tgt_vocab = len(lbl2id_map.keys())
    d_model = 512
    ocr_model = make_ocr_model(
        tgt_vocab, N=5, d_model=d_model, d_ff=2048, h=8, dropout=0.1)
    ocr_model.to(device)
    # --------------------------------------------------- #
    # load model parameters
    ocr_model.load_state_dict(torch.load(model_save_path))

    # ---------------------------------------------------------------- #
    # api for one img
    img_name = '201821285'
    img_path = os.path.join(base_data_dir, img_name + '.png')
    res = predict_student_num(img_path, id2lbl_map, ocr_model, sequence_len)
    print(res)
