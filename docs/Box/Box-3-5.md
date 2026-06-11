---
title: BOX3.5
comments: true  # 开启评论
---
对任意策略$\pi$，定义

$$r_\pi=[\ldots,r_\pi(s),\ldots]^T,$$

其中

$$r_\pi(s)=\sum_{a\in\mathcal{A}}\pi(a|s)\sum_{r\in\mathcal{R}}p(r|s,a)r,\quad s\in\mathcal{S}.$$

如果奖励发生仿射变换$r\to \alpha r+\beta$，那么

$$r_\pi(s)\to \alpha r_\pi(s)+\beta,$$

因此

$$r_\pi\to \alpha r_\pi+\beta\mathbf{1},$$

其中$\mathbf{1}=[1,\ldots,1]^T$。此时，贝尔曼最优方程变为

$$v'=\max_{\pi\in\Pi}(\alpha r_\pi+\beta\mathbf{1}+\gamma P_\pi v').\tag{3.9}$$

下面证明

$$v'=\alpha v^*+c\mathbf{1},\qquad c=\frac{\beta}{1-\gamma}$$

是式$(3.9)$的解。将$v'=\alpha v^*+c\mathbf{1}$代入$(3.9)$可得

$$\begin{aligned}
\alpha v^*+c\mathbf{1}
&=\max_{\pi\in\Pi}(\alpha r_\pi+\beta\mathbf{1}+\gamma P_\pi(\alpha v^*+c\mathbf{1}))\\
&=\max_{\pi\in\Pi}(\alpha r_\pi+\beta\mathbf{1}+\alpha\gamma P_\pi v^*+c\gamma\mathbf{1}),
\end{aligned}$$

其中最后一个等号是因为$P_\pi\mathbf{1}=\mathbf{1}$。上式可整理为

$$\alpha v^*=\max_{\pi\in\Pi}(\alpha r_\pi+\alpha\gamma P_\pi v^*)+\beta\mathbf{1}+c\gamma\mathbf{1}-c\mathbf{1}.$$

因此只需满足

$$\beta\mathbf{1}+c\gamma\mathbf{1}-c\mathbf{1}=0.$$

由于$c=\beta/(1-\gamma)$，所以上式成立。因此

$$v'=\alpha v^*+c\mathbf{1}$$

确实是式$(3.9)$的解。又因为$(3.9)$仍然是贝尔曼最优方程，所以$v'$也是唯一解。

最后，由于$v'$是$v^*$的仿射变换，行动值之间的相对大小关系保持不变。因此，由$v'$得到的贪婪最优策略与由$v^*$得到的贪婪最优策略相同，即

$$\arg\max_{\pi\in\Pi}(r_\pi+\gamma P_\pi v')$$

与

$$\arg\max_{\pi\in\Pi}(r_\pi+\gamma P_\pi v^*)$$

得到的最优策略相同。

!!! note
    这个证明说明：在$\alpha>0$时，对所有奖励同时做同样的线性缩放和平移，不会改变“哪个动作更好”的相对排序，因此不会改变最优策略。

---
