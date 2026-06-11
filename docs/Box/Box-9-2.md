---
title: BOX9.2
comments: true  # 开启评论
---
首先，对任意$s\in\mathcal{S}$，有

$$\begin{aligned}
\nabla_\theta v_\pi(s)
&=\nabla_\theta\left[\sum_{a\in\mathcal{A}}\pi(a|s,\theta)q_\pi(s,a)\right]\\
&=\sum_{a\in\mathcal{A}}\left[\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)+\pi(a|s,\theta)\nabla_\theta q_\pi(s,a)\right].
\end{aligned}\tag{9.15}$$

其中行动值为

$$q_\pi(s,a)=r(s,a)+\gamma\sum_{s'\in\mathcal{S}}p(s'|s,a)v_\pi(s').$$

由于$r(s,a)=\sum_r rp(r|s,a)$与$\theta$无关，因此

$$\nabla_\theta q_\pi(s,a)=\gamma\sum_{s'\in\mathcal{S}}p(s'|s,a)\nabla_\theta v_\pi(s').$$

代入式$(9.15)$可得

$$\begin{aligned}
\nabla_\theta v_\pi(s)
&=\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)\\
&\quad+\gamma\sum_{a\in\mathcal{A}}\pi(a|s,\theta)\sum_{s'\in\mathcal{S}}p(s'|s,a)\nabla_\theta v_\pi(s').
\end{aligned}\tag{9.16}$$

令

$$u(s)\doteq \sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a).$$

又因为

$$\sum_{a\in\mathcal{A}}\pi(a|s,\theta)\sum_{s'\in\mathcal{S}}p(s'|s,a)\nabla_\theta v_\pi(s')
=\sum_{s'\in\mathcal{S}}[P_\pi]_{ss'}\nabla_\theta v_\pi(s'),$$

所以式$(9.16)$可以写成矩阵-向量形式：

$$\nabla_\theta v_\pi=u+\gamma(P_\pi\otimes I_m)\nabla_\theta v_\pi,$$

其中$n=|\mathcal{S}|$，$m$为参数向量$\theta$的维度，$\otimes$为Kronecker积。解这个线性方程可得

$$\begin{aligned}
\nabla_\theta v_\pi
&=(I_{nm}-\gamma P_\pi\otimes I_m)^{-1}u\\
&=[(I_n-\gamma P_\pi)^{-1}\otimes I_m]u.
\end{aligned}\tag{9.17}$$

因此，对任意状态$s$，

$$\begin{aligned}
\nabla_\theta v_\pi(s)
&=\sum_{s'\in\mathcal{S}}[(I_n-\gamma P_\pi)^{-1}]_{ss'}u(s')\\
&=\sum_{s'\in\mathcal{S}}[(I_n-\gamma P_\pi)^{-1}]_{ss'}
\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s',\theta)q_\pi(s',a).
\end{aligned}\tag{9.18}$$

接下来解释$[(I_n-\gamma P_\pi)^{-1}]_{ss'}$的概率含义。由于

$$ (I_n-\gamma P_\pi)^{-1}=I+\gamma P_\pi+\gamma^2P_\pi^2+\cdots,$$

所以

$$[(I_n-\gamma P_\pi)^{-1}]_{ss'}
=\sum_{k=0}^{\infty}\gamma^k[P_\pi^k]_{ss'}.$$

而$[P_\pi^k]_{ss'}$表示从$s$恰好经过$k$步转移到$s'$的概率。因此，$[(I_n-\gamma P_\pi)^{-1}]_{ss'}$就是从$s$出发，以任意步数转移到$s'$的折扣总概率。记

$$[(I_n-\gamma P_\pi)^{-1}]_{ss'}\doteq \Pr_\pi(s'|s),$$

则式$(9.18)$即为式$(9.14)$。

---
