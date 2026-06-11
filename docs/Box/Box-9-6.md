---
title: BOX9.6
comments: true  # 开启评论
---
首先，由

$$v_\pi(s)=\sum_{a\in\mathcal{A}}\pi(a|s,\theta)q_\pi(s,a)$$

可得

$$\begin{aligned}
\nabla_\theta v_\pi(s)
&=\nabla_\theta\left[\sum_{a\in\mathcal{A}}\pi(a|s,\theta)q_\pi(s,a)\right]\\
&=\sum_{a\in\mathcal{A}}\left[\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)+\pi(a|s,\theta)\nabla_\theta q_\pi(s,a)\right].
\end{aligned}\tag{9.29}$$

其中行动值满足

$$\begin{aligned}
q_\pi(s,a)
&=\sum_r p(r|s,a)(r-\bar{r}_\pi)+\sum_{s'}p(s'|s,a)v_\pi(s')\\
&=r(s,a)-\bar{r}_\pi+\sum_{s'}p(s'|s,a)v_\pi(s').
\end{aligned}$$

由于$r(s,a)=\sum_r rp(r|s,a)$与$\theta$无关，因此

$$\nabla_\theta q_\pi(s,a)
=-\nabla_\theta\bar{r}_\pi+\sum_{s'\in\mathcal{S}}p(s'|s,a)\nabla_\theta v_\pi(s').$$

代入式$(9.29)$可得

$$\begin{aligned}
\nabla_\theta v_\pi(s)
&=\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)
-\nabla_\theta\bar{r}_\pi\\
&\quad+\sum_{a\in\mathcal{A}}\pi(a|s,\theta)\sum_{s'\in\mathcal{S}}p(s'|s,a)\nabla_\theta v_\pi(s').
\end{aligned}\tag{9.30}$$

令

$$u(s)\doteq \sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a).$$

由于

$$\sum_{a\in\mathcal{A}}\pi(a|s,\theta)\sum_{s'\in\mathcal{S}}p(s'|s,a)\nabla_\theta v_\pi(s')
=\sum_{s'\in\mathcal{S}}p(s'|s)\nabla_\theta v_\pi(s'),$$

式$(9.30)$可写成矩阵-向量形式：

$$\nabla_\theta v_\pi=u-\mathbf{1}_n\otimes\nabla_\theta\bar{r}_\pi+(P_\pi\otimes I_m)\nabla_\theta v_\pi.$$

因此

$$\mathbf{1}_n\otimes\nabla_\theta\bar{r}_\pi
=u+(P_\pi\otimes I_m)\nabla_\theta v_\pi-\nabla_\theta v_\pi.$$

在上式两边左乘$d_\pi^T\otimes I_m$，可得

$$\begin{aligned}
(d_\pi^T\mathbf{1}_n)\otimes\nabla_\theta\bar{r}_\pi
&=(d_\pi^T\otimes I_m)u+(d_\pi^TP_\pi\otimes I_m)\nabla_\theta v_\pi-(d_\pi^T\otimes I_m)\nabla_\theta v_\pi\\
&=(d_\pi^T\otimes I_m)u,
\end{aligned}$$

其中使用了$d_\pi^T\mathbf{1}_n=1$和$d_\pi^TP_\pi=d_\pi^T$。于是

$$\begin{aligned}
\nabla_\theta\bar{r}_\pi
&=(d_\pi^T\otimes I_m)u\\
&=\sum_{s\in\mathcal{S}}d_\pi(s)u(s)\\
&=\sum_{s\in\mathcal{S}}d_\pi(s)\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a).
\end{aligned}$$

最后使用

$$\nabla_\theta\pi(a|s,\theta)=\pi(a|s,\theta)\nabla_\theta\ln\pi(a|s,\theta),$$

即可得到

$$\nabla_\theta\bar{r}_\pi
=\mathbb{E}[\nabla_\theta\ln\pi(A|S,\theta)q_\pi(S,A)],$$

其中$S\sim d_\pi$，$A\sim\pi(S,\theta)$。

---
