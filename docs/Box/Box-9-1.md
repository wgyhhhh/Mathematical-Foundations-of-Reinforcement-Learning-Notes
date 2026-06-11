---
title: BOX9.1
comments: true  # 开启评论
---
**第一步：证明对任意初始状态$s_0\in\mathcal{S}$，都有**

$$\bar{r}_\pi=\lim_{n\to\infty}\mathbb{E}\left[\frac{1}{n}\sum_{t=0}^{n-1}R_{t+1}\middle|S_0=s_0\right].\tag{9.6}$$

注意到

$$\begin{aligned}
\lim_{n\to\infty}\mathbb{E}\left[\frac{1}{n}\sum_{t=0}^{n-1}R_{t+1}\middle|S_0=s_0\right]
&=\lim_{n\to\infty}\frac{1}{n}\sum_{t=0}^{n-1}\mathbb{E}[R_{t+1}|S_0=s_0]\\
&=\lim_{t\to\infty}\mathbb{E}[R_{t+1}|S_0=s_0].
\end{aligned}\tag{9.7}$$

最后一个等号来自Cesaro均值的性质：如果序列$\{a_k\}$收敛，则其前$n$项平均值也收敛到同一个极限。

下面进一步分析$\mathbb{E}[R_{t+1}|S_0=s_0]$。由全期望公式，

$$\begin{aligned}
\mathbb{E}[R_{t+1}|S_0=s_0]
&=\sum_{s\in\mathcal{S}}\mathbb{E}[R_{t+1}|S_t=s,S_0=s_0]p^{(t)}(s|s_0)\\
&=\sum_{s\in\mathcal{S}}\mathbb{E}[R_{t+1}|S_t=s]p^{(t)}(s|s_0)\\
&=\sum_{s\in\mathcal{S}}r_\pi(s)p^{(t)}(s|s_0),
\end{aligned}$$

其中$p^{(t)}(s|s_0)$表示从$s_0$恰好经过$t$步转移到$s$的概率。第二个等号来自马尔科夫无记忆性质：下一步奖励只依赖当前状态，而不依赖更早的历史。

由稳态分布定义，

$$\lim_{t\to\infty}p^{(t)}(s|s_0)=d_\pi(s).$$

因此初始状态$s_0$不会影响长期结果，并且

$$\lim_{t\to\infty}\mathbb{E}[R_{t+1}|S_0=s_0]
=\sum_{s\in\mathcal{S}}r_\pi(s)d_\pi(s)=\bar{r}_\pi.$$

将其代入式$(9.7)$，即可得到式$(9.6)$。

**第二步：考虑任意初始状态分布$d$。**

由全期望公式，

$$\begin{aligned}
\lim_{n\to\infty}\mathbb{E}\left[\frac{1}{n}\sum_{t=0}^{n-1}R_{t+1}\right]
&=\lim_{n\to\infty}\frac{1}{n}\sum_{s\in\mathcal{S}}d(s)\mathbb{E}\left[\sum_{t=0}^{n-1}R_{t+1}\middle|S_0=s\right]\\
&=\sum_{s\in\mathcal{S}}d(s)\lim_{n\to\infty}\mathbb{E}\left[\frac{1}{n}\sum_{t=0}^{n-1}R_{t+1}\middle|S_0=s\right].
\end{aligned}$$

由于式$(9.6)$对任意初始状态都成立，所以

$$\lim_{n\to\infty}\mathbb{E}\left[\frac{1}{n}\sum_{t=0}^{n-1}R_{t+1}\right]
=\sum_{s\in\mathcal{S}}d(s)\bar{r}_\pi=\bar{r}_\pi.$$

证明完毕。

---
