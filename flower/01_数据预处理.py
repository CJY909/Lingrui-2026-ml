# -*- coding: utf-8 -*-
# 数据预处理：1.清理损坏图片 2.划分数据集 3.数据增强 4.统一图片尺寸
import os
import shutil
from PIL import Image

# 脚本放在哪个文件夹，flower 数据集就默认在它旁边
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, 'flower')          # 原始数据集
DST = os.path.join(BASE, 'flower_data')     # 处理完的数据集
BAD = os.path.join(BASE, 'bad_images')      # 损坏图片统一挪到这里，不直接删，保险一点


# ---------- 第一步：检查并清理损坏的图片 ----------
# 注意：不能只查 .jpg，数据集里还混了少量 .png（有的其实是坏文件）
# 顺便把 png 统一转成 jpg，保证整个数据集格式一致
def check_images():
    bad = []
    total = 0
    for root, _, files in os.walk(SRC):
        for name in files:
            if not name.lower().endswith(('.jpg', '.png')):
                continue
            total += 1
            path = os.path.join(root, name)
            try:
                # verify() 只检查文件头，im.load() 会真正解码像素，两步都过才算好图
                img = Image.open(path)
                img.verify()
                img = Image.open(path)
                img.load()
                # 有效但不是jpg的，转存成jpg
                if not name.lower().endswith('.jpg'):
                    img.convert('RGB').save(path[:-4] + '.jpg', 'JPEG')
                    os.remove(path)
            except Exception:
                bad.append(path)
    return total, bad


def remove_bad(bad_list):
    os.makedirs(BAD, exist_ok=True)
    for p in bad_list:
        # 挪走之前先保证目标文件夹里没有同名文件
        target = os.path.join(BAD, os.path.basename(p))
        if os.path.exists(target):
            target = target.replace('.jpg', '_1.jpg')
        shutil.move(p, target)
    print('损坏图片数量: %d，已全部移动到 bad_images 文件夹' % len(bad_list))


# ---------- 第二步：划分数据集 ----------
# 官方只给了 train 和 test，我把 train 再切成 80% 训练 + 20% 验证
# 验证集用来在训练过程中看模型表现，test 留到最后测
def split_dataset(ratio=0.8):
    if os.path.exists(DST):
        shutil.rmtree(DST)  # 重复运行时先清空，避免残留旧数据
    classes = os.listdir(os.path.join(SRC, 'train'))
    for cls in classes:
        imgs = os.listdir(os.path.join(SRC, 'train', cls))
        # 打乱顺序再切，保证每一类都是随机抽的
        import random
        random.seed(2024)
        random.shuffle(imgs)
        n_train = int(len(imgs) * ratio)
        for i, name in enumerate(imgs):
            sub = 'train' if i < n_train else 'val'
            new_dir = os.path.join(DST, sub, cls)
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy(os.path.join(SRC, 'train', cls, name),
                        os.path.join(new_dir, name))
        # test 集直接原样拷过来
        new_dir = os.path.join(DST, 'test', cls)
        os.makedirs(new_dir, exist_ok=True)
        for name in os.listdir(os.path.join(SRC, 'test', cls)):
            shutil.copy(os.path.join(SRC, 'test', cls, name),
                        os.path.join(new_dir, name))
    # 打印一下每个集合的数量
    for sub in ['train', 'val', 'test']:
        cnt = sum(len(os.listdir(os.path.join(DST, sub, c))) for c in classes)
        print('%s: %d 张' % (sub, cnt))


if __name__ == '__main__':
    total, bad = check_images()
    print('共扫描 jpg 图片 %d 张' % total)
    if bad:
        for p in bad:
            print('  损坏: %s' % p)
        remove_bad(bad)
    else:
        print('没有发现损坏图片')
    split_dataset()
    print('预处理完成，处理后的数据集保存在 flower_data 文件夹')
