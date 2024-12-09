import cv2
import numpy as np
from flask import Flask, render_template

# 读取图像
dog_img = cv2.imread('dog.jpg')  # 读取狗的图片
cat_img = cv2.imread('cat.jpg')  # 读取猫的图片

# 确保图像尺寸一致
dog_img = cv2.resize(dog_img, (512, 512))
cat_img = cv2.resize(cat_img, (512, 512))

# 创建高斯金字塔
def gaussian_pyramid(img, levels=6):
    pyramid = [img]
    for i in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid

# 创建拉普拉斯金字塔
def laplacian_pyramid(gaussian_pyramid):
    pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
        pyramid.append(laplacian)
    pyramid.append(gaussian_pyramid[-1])  # 顶层是高斯金字塔的最后一层
    return pyramid

# 融合金字塔
def fuse_pyramids(dog_pyr, cat_pyr):
    fused_pyramid = []
    for dog_layer, cat_layer in zip(dog_pyr, cat_pyr):
        fused_layer = cv2.addWeighted(dog_layer, 0.5, cat_layer, 0.5, 0)
        fused_pyramid.append(fused_layer)
    return fused_pyramid

# 重建图像
def reconstruct_from_pyramid(pyramid):
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        img = cv2.pyrUp(img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        img = cv2.add(img, pyramid[i])
    return img

# 创建金字塔
dog_gaussian = gaussian_pyramid(dog_img)
cat_gaussian = gaussian_pyramid(cat_img)

dog_laplacian = laplacian_pyramid(dog_gaussian)
cat_laplacian = laplacian_pyramid(cat_gaussian)

# 融合拉普拉斯金字塔
fused_pyramid = fuse_pyramids(dog_laplacian, cat_laplacian)

# 从金字塔重建融合图像
fused_image = reconstruct_from_pyramid(fused_pyramid)

# 保存结果
cv2.imwrite('static/processed_pyramids/fused_image.jpg', fused_image)

# 保存金字塔每一层的图像
for i, (dog_layer, cat_layer, fused_layer) in enumerate(zip(dog_laplacian, cat_laplacian, fused_pyramid)):
    cv2.imwrite(f'static/processed_pyramids/dog_layer_{i}.jpg', dog_layer)
    cv2.imwrite(f'static/processed_pyramids/cat_layer_{i}.jpg', cat_layer)
    cv2.imwrite(f'static/processed_pyramids/fused_layer_{i}.jpg', fused_layer)

# Flask 应用程序
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
