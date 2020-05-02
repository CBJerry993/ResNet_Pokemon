# ResNet18识别宝可梦数据集

版本：v1.0.20200502

## 数据集介绍

[数据集官网](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/)

数据集在文件夹pokemon内，共有5大分类，1168张图片。包含了妙蛙种子234张、小火龙238张、杰尼龟223张、皮卡丘234张、超梦239张。

![](https://upload-images.jianshu.io/upload_images/19723859-8cd090015c847719.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 项目介绍

模型使用的是ResNet18，训练了10个epochs，使用visdom可视化准确率和损失。最终对比了迁移学习和自定义的残差网络效果，迁移学习效果更佳。项目主要文件如下：

- pokemon.py 用于加载和标记数据集
- resnet.py 自己写的残差网络
- train_scratch.py 使用自己写的残差网络训练
- train_transfer.py 使用已训练过的残差网络训练（迁移学习）
- utils.py 一些辅助方法（打平数据和显示图像）
- best_scratch.mdl / best_transfer.mdl 最佳准确率的参数，有则该无则增。可以在train之前删掉。

## 项目流程

### 预处理

预处理主要有图片的resize操作，数据增强，标准化和转换成tensor，加载标记和切分数据集。

### 自定义残差网络

### 迁移学习残差网络

### 训练和测试

### 训练结果

train_scratch是自己搭建的ResNet18，train_transfer是使用已经训练好的ResNet18进行迁移学习。后者效果更佳！

| 训练文件       | 最佳准确率 best acc | 测试集准确率 test acc |
| -------------- | ------------------- | --------------------- |
| train_scratch  | 89.6%               | 92%                   |
| train_transfer | 94.8%               | 95.2%                 |

![](https://upload-images.jianshu.io/upload_images/19723859-9b6c3e2d101f38a7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/19723859-7faa95df19e96fe0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 项目总结

虽然整体的数据集较小（1168张图片），但是迁移学习比自定义的ResNet效果更佳，准确率可以达到95%+。

其他待研究：增加训练epochs，修改学习率，数据增强等操作能否再提高准确率？