import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

# 定义训练和验证时的transforms
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/class_image_transforms.html
train_transforms = pdx.transforms.Compose([
    pdx.transforms.RandomCrop(crop_size=224),
    pdx.transforms.RandomHorizontalFlip(),
    pdx.transforms.Normalize()
])

eval_transforms = pdx.transforms.Compose([
    pdx.transforms.ResizeByShort(short_size=256),
    pdx.transforms.CenterCrop(crop_size=224),
    pdx.transforms.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets/classification.html#classificationdataset
train_dataset = pdx.datasets.ImageNet(
    data_dir='images',
    file_list='images/train_list.txt',
    label_list='images/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.ImageNet(
    data_dir='images',
    file_list='Images/val_list.txt',
    label_list='Images/labels.txt',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/classification.html#paddlex-classification
model = pdx.cls.MobileNetV3_small_ssld(num_classes=len(train_dataset.labels))

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/classification.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=2,
    train_dataset=train_dataset,
    train_batch_size=64,
    eval_dataset=eval_dataset,
    lr_decay_epochs=[4, 6, 8],
    learning_rate=0.025,
    save_dir='output/MobileNetV3_small_ssld',
    save_interval_epochs=1,
    log_interval_steps=100,
    use_vdl=True)
import cv2
import matplotlib.pyplot as plt
# 加载彩色图
img = cv2.imread('images/CAKE/CAKE0000.png', 1)
# 将彩色图的BGR通道顺序转成RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 显示图片
plt.imshow(img)


import cv2
import matplotlib.pyplot as plt
# 加载彩色图
img = cv2.imread('images/BEANS/BEANS0000.png', 1)
# 将彩色图的BGR通道顺序转成RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 显示图片
plt.imshow(img)
test_jpg = 'images/BEANS/BEANS0000.png'
model = pdx.load_model('output/MobileNetV3_small_ssld/pretrain')
result = model.predict(test_jpg)
print("Predict Result: ", result)