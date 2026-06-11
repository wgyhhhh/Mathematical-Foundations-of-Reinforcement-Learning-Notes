---
title: BOX5.1
comments: true  # 开启评论
---
对随机变量$X$，假设$\{x_i\}_{i=1}^n$是一些独立同分布样本。令

$$\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i$$

表示这些样本的均值。那么有

$$\mathbb{E}[\bar{x}]=\mathbb{E}[X],$$

以及

$$\mathrm{var}[\bar{x}]=\frac{1}{n}\mathrm{var}[X].$$

这两个等式说明：$\bar{x}$是$\mathbb{E}[X]$的无偏估计，并且当$n\to\infty$时，其方差会逐渐减小到$0$。

证明如下。首先，

$$\mathbb{E}[\bar{x}]
=\mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n x_i\right]
=\frac{1}{n}\sum_{i=1}^n\mathbb{E}[x_i]
=\mathbb{E}[X].$$

最后一个等号成立，是因为样本是同分布的，即$\mathbb{E}[x_i]=\mathbb{E}[X]$。

其次，

$$\mathrm{var}(\bar{x})
=\mathrm{var}\left(\frac{1}{n}\sum_{i=1}^{n}x_i\right)
=\frac{1}{n^2}\sum_{i=1}^{n}\mathrm{var}[x_i]
=\frac{n\cdot \mathrm{var}[X]}{n^2}
=\frac{1}{n}\mathrm{var}[X].$$

第二个等号成立，是因为样本之间相互独立；第三个等号成立，是因为样本是同分布的，即$\mathrm{var}[x_i]=\mathrm{var}[X]$。

---
