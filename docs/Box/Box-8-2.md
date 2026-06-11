---
title: BOX8.2
comments: true  # 开启评论
---
下面证明第7章式$(7.1)$中的表格型TD算法是式$(8.14)$中TD-Linear算法的一个特例。

对任意$s\in\mathcal{S}$，考虑如下特殊特征向量：

$$\phi(s)=e_s\in\mathbb{R}^n,$$

其中$e_s$是一个one-hot向量：与状态$s$对应的元素为$1$，其他元素为$0$。此时

$$\hat{v}(s,w)=e_s^Tw=w(s),$$

其中$w(s)$表示向量$w$中与状态$s$对应的元素。

将上式代入式$(8.14)$可得

$$w_{t+1}=w_t+\alpha_t\left(r_{t+1}+\gamma w_t(s_{t+1})-w_t(s_t)\right)e_{s_t}.$$

由于$e_{s_t}$的定义，上式只会更新$w_t(s_t)$这一项。于是，在等式两边同时左乘$e_{s_t}^T$，可得

$$w_{t+1}(s_t)=w_t(s_t)+\alpha_t\left(r_{t+1}+\gamma w_t(s_{t+1})-w_t(s_t)\right).$$

这正是第7章式$(7.1)$中的表格型TD算法。

因此，当特征向量选择为$\phi(s)=e_s$时，TD-Linear算法退化为表格型TD算法。

---
