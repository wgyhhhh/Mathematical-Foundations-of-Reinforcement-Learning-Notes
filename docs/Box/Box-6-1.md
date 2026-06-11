---
title: BOX6.1
comments: true  # 开启评论
---
下面证明SGD算法是RM算法的一种特殊形式。因此，SGD的收敛性可以由RM定理直接推出。

SGD要解决的问题是最小化

$$J(w)=\mathbb{E}[f(w,X)].$$

该优化问题可以转化为一个求根问题，即寻找

$$\nabla_wJ(w)=\mathbb{E}[\nabla_w f(w,X)]=0$$

的根。令

$$g(w)=\nabla_wJ(w)=\mathbb{E}[\nabla_w f(w,X)].$$

那么SGD的目标就是寻找$g(w)=0$的根，这正是RM算法要解决的问题。

实际中可以观测到的是

$$\tilde{g}(w,\eta)=\nabla_w f(w,x),$$

其中$x$是随机变量$X$的一个样本。注意$\tilde{g}$可以写成

$$\begin{aligned}
\tilde{g}(w,\eta)
&=\nabla_w f(w,x)\\
&=\mathbb{E}[\nabla_w f(w,X)]
+\underbrace{\nabla_w f(w,x)-\mathbb{E}[\nabla_w f(w,X)]}_{\eta}.
\end{aligned}$$

因此，用RM算法求解$g(w)=0$得到

$$w_{k+1}=w_k-a_k\tilde{g}(w_k,\eta_k)=w_k-a_k\nabla_w f(w_k,x_k),$$

这与式$(6.13)$中的SGD算法完全一致。所以SGD算法是RM算法的一个特例。

接下来证明定理6.1中的三个条件在这里成立。

1. 因为

    $$\nabla_wg(w)=\nabla_w\mathbb{E}[\nabla_w f(w,X)]=\mathbb{E}[\nabla_w^2 f(w,X)],$$

    由条件$c_1\leq\nabla_w^2f(w,X)\leq c_2$可得

    $$c_1\leq\nabla_wg(w)\leq c_2.$$

    因此，定理6.1中的第一个条件成立。

2. 定理6.1中的第二个条件与定理6.4中的第二个条件相同，因此直接成立。

3. 定理6.1中的第三个条件要求

    $$\mathbb{E}[\eta_k|\mathcal{H}_k]=0,\qquad \mathbb{E}[\eta_k^2|\mathcal{H}_k]<\infty.$$

    由于$\{x_k\}$是独立同分布的，对所有$k$都有

    $$\mathbb{E}_{x_k}[\nabla_w f(w,x_k)]=\mathbb{E}[\nabla_w f(w,X)].$$

    因此

    $$\mathbb{E}[\eta_k|\mathcal{H}_k]
    =\mathbb{E}[\nabla_w f(w_k,x_k)-\mathbb{E}[\nabla_w f(w_k,X)]|\mathcal{H}_k].$$

    因为$\mathcal{H}_k=\{w_k,w_{k-1},\ldots\}$，且$x_k$与$\mathcal{H}_k$独立，所以右侧第一项为

    $$\mathbb{E}[\nabla_w f(w_k,x_k)|\mathcal{H}_k]=\mathbb{E}_{x_k}[\nabla_w f(w_k,x_k)].$$

    第二项为

    $$\mathbb{E}[\mathbb{E}[\nabla_w f(w_k,X)]|\mathcal{H}_k]=\mathbb{E}[\nabla_w f(w_k,X)],$$

    因为$\mathbb{E}[\nabla_w f(w_k,X)]$是$w_k$的函数。于是

    $$\mathbb{E}[\eta_k|\mathcal{H}_k]
    =\mathbb{E}_{x_k}[\nabla_w f(w_k,x_k)]-\mathbb{E}[\nabla_w f(w_k,X)]=0.$$

    类似地，如果对任意$x$和任意$w$都有$|\nabla_w f(w,x)|<\infty$，则可证明

    $$\mathbb{E}[\eta_k^2|\mathcal{H}_k]<\infty.$$

综上，定理6.1中的三个条件均满足，因此SGD算法收敛。

---
