---
title: BOX3.4
comments: true  # 开启评论
---
虽然最优策略的矩阵-向量形式为

$$\pi^*=\arg\max_\pi(r_\pi+\gamma P_\pi v^*),$$

但它的逐元素形式为

$$
\pi^*(s)=\arg\max_{\pi\in\Pi}\sum_{a\in\mathcal{A}}\pi(a|s)
\underbrace{\left(\sum_{r\in\mathcal{R}}p(r|s,a)r+\gamma\sum_{s'\in\mathcal{S}}p(s'|s,a)v^*(s')\right)}_{q^*(s,a)},\quad s\in\mathcal{S}.
$$

也就是说，对每个状态$s$，策略$\pi(s)$需要最大化

$$\sum_{a\in\mathcal{A}}\pi(a|s)q^*(s,a).$$

由于$\pi(a|s)$是关于行动$a$的概率分布，上式是各个$q^*(s,a)$的加权平均值。显然，当策略把全部概率分配给$q^*(s,a)$最大的行动时，该加权平均值取得最大。

因此，确定性贪婪策略

$$
\pi^*(a|s)=
\begin{cases}
1,& a=a^*(s),\\
0,& a\neq a^*(s),
\end{cases}
$$

其中

$$a^*(s)=\arg\max_a q^*(s,a),$$

就是一个最优策略。

!!! note
    这里的关键直觉是：在固定状态$s$下，$\sum_a\pi(a|s)q^*(s,a)$只是对不同动作价值的加权平均。加权平均不可能超过最大值本身，所以把概率全部放到最大动作价值对应的动作上即可。

---
