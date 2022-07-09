import os
import argparse
import cv2
from tqdm import tqdm
from pyimgsaliency import binarise_saliency_map, get_saliency_mbd, get_saliency_rbd, get_saliency_ft


def get_saliency_map(filename, algo_name='rbd', is_bin=False, img_size=None):
    if algo_name == 'rbd':
        saliency_map = get_saliency_rbd(filename, img_size=img_size).astype('uint8')
    elif algo_name == 'mbd':
        saliency_map = get_saliency_mbd(filename, img_size=img_size).astype('uint8')
    elif algo_name == 'ft':
        saliency_map = get_saliency_ft(filename, img_size=img_size).astype('uint8')
    else:
        print('algorithm input error!')
        exit()

    if is_bin:
        binary_sal = binarise_saliency_map(saliency_map, method='adaptive')
        saliency_map = 255 * binary_sal.astype('uint8')

    return saliency_map


def test(args):
     # path to the image
    # filename = './images/bird.jpg'
    filename = './images/000012.jpg'

    img = cv2.imread(filename)
    if args.img_size is not None:
        img = cv2.resize(img, (args.img_size, args.img_size))
    saliency_map = get_saliency_map(filename, algo_name=args.algo_name, is_bin=args.is_bin, img_size=args.img_size)

    cv2.imshow('img', img)
    cv2.imshow(args.algo_name + '-binary' if args.is_bin else args.algo_name, saliency_map)

    cv2.waitKey(0)


def build_saliency_dataset(args):
    cod10k_categories = os.listdir(args.image_dir)
    category_num = len(cod10k_categories)
    

    if not os.path.exists(args.dir_to_save_saliency):
        os.makedirs(args.dir_to_save_saliency)
    
    start_idx, end_idx = 0, category_num
    # for category in cod10k_categories:
    for cat_idx in range(start_idx, end_idx):
        category = cod10k_categories[cat_idx]
        print('processing the {} category: {}'.format(cat_idx, category))
        # 指定类别的saliency文件夹
        if not os.path.exists(os.path.join(args.dir_to_save_saliency, category)):
            os.makedirs(os.path.join(args.dir_to_save_saliency, category))

        # 处理指定类别的所有图像
        imagenames = os.listdir(os.path.join(args.image_dir, category))
        with tqdm(total=len(imagenames)) as pbar:
            for imagename in imagenames:
                if imagename[-3:] == 'png': # png格式的图像为4通道
                    continue
                img_path = os.path.join(args.image_dir, category, imagename)
                try:
                    saliency_map = get_saliency_map(img_path, algo_name=args.algo_name, is_bin=args.is_bin, img_size=args.img_size)
                except Exception as e:
                    print('{}: {}'.format(category, imagename))
                    continue
                path_to_save_saliency = os.path.join(args.dir_to_save_saliency, category, imagename)
                cv2.imwrite(path_to_save_saliency, saliency_map)
                pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract saliency map')
    parser.add_argument('--image_dir', type=str, default='./dataset')
    parser.add_argument('--dir_to_save_saliency', type=str, default='F:/saliency/')
    
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--algo_name', type=str, default='rbd')
    parser.add_argument('--is_bin', type=bool, default=False)   # 是否进行二值化处理

    args = parser.parse_args()

    # test(args)
    build_saliency_dataset(args)
    
    
