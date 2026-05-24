"""
案例:
    演示基础图像操作

图像分类:
    二值图:        1通道, 每个像素点由0,1组成
    灰度图:        1通道, 每个像素点的范围: [0, 255]
    索引图:        1通道, 每个像素点的范围: [0, 255], 像素点表示像素的索引
    RGB真彩图:     3通道, Red, Green, Blue, 红蓝绿

涉及到的API:
    imshow()    基于HWC, 展示图片
    imread()    读取图片, 获取HWC
    imsave()    基于HWC, 保存图片
"""

# 导包
import torch
import matplotlib.pyplot as plt
import numpy as np

# 1.定义函数, 绘制: 全黑, 全白图
def dm01():
    # 1.定义全黑图片: 像素点越接近0越黑, 越接近255越白
    # HWC: 高度, 宽度, 通道数
    img1 = np.zeros((200, 200, 3))
    print(f'img1: {img1}')

    # 2.绘制图片
    plt.imshow(img1)
    plt.show()

    # 3.定义全白图片
    img2 = torch.full(size=(200, 200, 3), fill_value=255)
    print(f'img2: {img2}')
    plt.imshow(img2)
    plt.show()


# 2.定义函数, 加载图片
def dm02():
    # 1.加载图片
    img = plt.imread('./data/img.jpg')
    print(f'img: {img}')
    print(f'img.shape: {img.shape}')

    # 2.保存图片
    plt.imsave('./data/img_copy.jpg', img)

    # 3.展示图片
    plt.imshow(img)
    plt.show()

# 测试
if __name__ == '__main__':
    # dm01()
    dm02()