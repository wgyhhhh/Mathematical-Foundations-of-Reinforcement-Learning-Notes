---
title: BOX4.3
comments: true  # 开启评论
---
首先，由于

$$v_{\pi_k}^{(j)}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}^{(j-1)},$$

以及

$$v_{\pi_k}^{(j+1)}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}^{(j)},$$

所以有

$$v_{\pi_k}^{(j+1)}-v_{\pi_k}^{(j)}
=\gamma P_{\pi_k}\left(v_{\pi_k}^{(j)}-v_{\pi_k}^{(j-1)}\right)
=\cdots
=\gamma^jP_{\pi_k}^j\left(v_{\pi_k}^{(1)}-v_{\pi_k}^{(0)}\right).\tag{4.5}$$

其次，由于$v_{\pi_k}^{(0)}=v_{\pi_{k-1}}$，因此

$$\begin{aligned}
v_{\pi_k}^{(1)}
&=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}^{(0)}\\
&=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_{k-1}}\\
&\geq r_{\pi_{k-1}}+\gamma P_{\pi_{k-1}}v_{\pi_{k-1}}\\
&=v_{\pi_{k-1}}=v_{\pi_k}^{(0)}.
\end{aligned}$$

其中不等式来自

$$\pi_k=\arg\max_\pi(r_\pi+\gamma P_\pi v_{\pi_{k-1}}).$$

将$v_{\pi_k}^{(1)}\geq v_{\pi_k}^{(0)}$代入式$(4.5)$，并利用$P_{\pi_k}$为非负矩阵，可得

$$v_{\pi_k}^{(j+1)}\geq v_{\pi_k}^{(j)}.$$

这说明在截断策略迭代中，策略评估的多步迭代会逐步提高状态值估计。

---
