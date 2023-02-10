## 4小鼠实验动物Classification

这是一个自制的数据集，裁切自MOT，来源于真实的4只C57小鼠社交实验，不仅四至小鼠长得一模一样，有时候挤在一起形变和干扰非常多，难度可以说是相当变态了。

|mouse1|mouse2|mouse3|mouse4|
|-|-|-|-|
|![1](https://github.com/tctco/mmlab/blob/main/classification-mice/imgs/mouse1.png?raw=true)|![2](https://github.com/tctco/mmlab/blob/main/classification-mice/imgs/mouse2.png?raw=true)|![3](https://github.com/tctco/mmlab/blob/main/classification-mice/imgs/mouse3.png?raw=true)|![4](https://github.com/tctco/mmlab/blob/main/classification-mice/imgs/mouse4.png?raw=true)|

使用`hornet-tiny`和`densenet121`，相对来说前者看上去更稳定一些（也可能与优化器、数据增强等其他因素有关）。DenseNet是[SIPEC](https://github.com/SIPEC-Animal-Data-Analysis/SIPEC)给出的解决方案，但是波动非常大，最高可以到54%，低的时候只有45%左右。

![acc_curve](https://raw.githubusercontent.com/tctco/mmlab/main/classification-mice/imgs/acc_curve.png)

最终在测试集的精度可以达到59%左右，感觉已经算是意外之喜了……

## 混淆矩阵

![confusion_matrix](https://raw.githubusercontent.com/tctco/mmlab/main/classification-mice/imgs/confusion_matrix.png)

## EigenGardCAM

小鼠的尾部确实做了一点标记。

![cam](https://raw.githubusercontent.com/tctco/mmlab/main/classification-mice/imgs/cam.png)

## Benchmark

在后续的测试中发现MobileNetV2表现比较不错。

|Hornet-tiny|Densenet101|MobilenetV2|EfficientnetB0|VAN|
|:-:|:-:|:-:|:-:|:-:|
|59|57|**68**|67|54|