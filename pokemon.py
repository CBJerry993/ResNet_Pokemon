import csv
import glob
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision


class Pokemon(Dataset):

    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()
        self.root = root
        self.resize = resize
        # 编码str的类型，使用dict映射。
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):  # 排序操作一下，避免每次加载尽量的顺序不一样。
            # 如果不是目录，过滤掉。
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())

        print(self.name2label)

        # image, label
        self.images, self.labels = self.load_csv('images.csv')

        # 裁剪训练集和测试集 6:2:2
        if mode == 'train':  # 60%
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':  # 20% = 60%->80%
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:  # 20% = 80%->100%
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def __len__(self):

        return len(self.images)  # 1167*0.6或者*0.2

    # 可视化的时候返回normalize之前的状态，做个逆操作
    def denormalize(self, x_hat):
        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)
        std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path => image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),  # 稍微设置大一点
            transforms.RandomRotation(15),  # 随机旋转-15~15
            transforms.CenterCrop(self.resize),  # 中心裁剪成resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 这些数据是统计大量的imagnet来的。
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label

    def load_csv(self, filename):
        # 如果不存在，才创建csv。
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                # 'pokemon\\mewtwo\\00001.png，类别信息mewtwo使用此路径来判定。
                # 图片可能有多种格式，使用glob通配符
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            print(len(images), images)  # 1167, 'pokemon\\bulbasaur\\00000000.png'
            # 打乱图片
            random.shuffle(images)
            # 把路径和图片的关系，存储到csv，这样使得及时图片在任何位置都可以匹配到。
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2]  # bulbasaur
                    label = self.name2label[name]
                    writer.writerow([img, label])  # 'pokemon\\bulbasaur\\00000000.png', 0
                print('writen into csv file:', filename)

        # 存在直接加载csv
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row  # 'pokemon\\bulbasaur\\00000000.png', 0
                label = int(label)
                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)  # 保证数据一致
        return images, labels


def main():
    import visdom
    import time

    # 规整写法
    viz = visdom.Visdom()  # 需要开启进程。 python3 -m visdom.server
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    db = torchvision.datasets.ImageFolder(root='pokemon', transform=tf)
    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=4)
    print(db.class_to_idx)  # {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}

    for x, y in loader:
        viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        time.sleep(5)

    # 通用写法
    # db = Pokemon('pokemon', 64, 'train')
    # x, y = next(iter(db))
    # print('sample:', x.shape, y.shape, y)  # sample: torch.Size([3, 64, 64]) torch.Size([]) tensor(1)
    #
    # # 逆操作显示
    # viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    #
    # # batch批加载
    # loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)
    #
    # for x, y in loader:
    #     viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))  # 一行显示8张，
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #
    #     time.sleep(5)


if __name__ == '__main__':
    main()
