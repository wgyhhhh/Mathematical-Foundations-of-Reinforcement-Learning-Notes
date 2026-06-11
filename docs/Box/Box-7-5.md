---
title: BOX7.5
comments: true  # 开启评论
---
根据期望的定义，式$(7.19)$可以改写为

$$q(s,a)=\sum_r p(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)\max_{a'\in\mathcal{A}(s')}q(s',a').$$

对等式两边同时关于$a\in\mathcal{A}(s)$取最大值，可得

$$\max_{a\in\mathcal{A}(s)}q(s,a)
=\max_{a\in\mathcal{A}(s)}
\left[
\sum_r p(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)\max_{a'\in\mathcal{A}(s')}q(s',a')
\right].
$$

记

$$v(s)\doteq \max_{a\in\mathcal{A}(s)}q(s,a),$$

则上式可改写为

$$\begin{aligned}
v(s)
&=\max_{a\in\mathcal{A}(s)}
\left[
\sum_r p(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v(s')
\right]\\
&=\max_{\pi}
\sum_{a\in\mathcal{A}(s)}\pi(a|s)
\left[
\sum_r p(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v(s')
\right].
\end{aligned}$$

这正是第3章中介绍的基于状态值表示的贝尔曼最优方程。因此，式$(7.19)$就是基于行动值表示的贝尔曼最优方程。

---
