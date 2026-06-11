---
title: BOX8.6
comments: true  # 开启评论
---
注意

$$\begin{aligned}
\|\Phi w^*-v_\pi\|_D
&=\|\Phi w^*-Mv_\pi+Mv_\pi-v_\pi\|_D\\
&\leq \|\Phi w^*-Mv_\pi\|_D+\|Mv_\pi-v_\pi\|_D\\
&=\|MT_\pi(\Phi w^*)-MT_\pi(v_\pi)\|_D+\|Mv_\pi-v_\pi\|_D.
\end{aligned}\tag{8.33}$$

最后一个等号成立，是因为

$$\Phi w^*=MT_\pi(\Phi w^*),\qquad v_\pi=T_\pi(v_\pi).$$

又有

$$\begin{aligned}
MT_\pi(\Phi w^*)-MT_\pi(v_\pi)
&=M(r_\pi+\gamma P_\pi\Phi w^*)-M(r_\pi+\gamma P_\pi v_\pi)\\
&=\gamma MP_\pi(\Phi w^*-v_\pi).
\end{aligned}$$

将其代入式$(8.33)$，可得

$$\begin{aligned}
\|\Phi w^*-v_\pi\|_D
&\leq \|\gamma MP_\pi(\Phi w^*-v_\pi)\|_D+\|Mv_\pi-v_\pi\|_D\\
&\leq \gamma\|M\|_D\|P_\pi(\Phi w^*-v_\pi)\|_D+\|Mv_\pi-v_\pi\|_D\\
&=\gamma\|P_\pi(\Phi w^*-v_\pi)\|_D+\|Mv_\pi-v_\pi\|_D\\
&\leq \gamma\|\Phi w^*-v_\pi\|_D+\|Mv_\pi-v_\pi\|_D.
\end{aligned}$$

其中用到了$\|M\|_D=1$和$\|P_\pi x\|_D\leq\|x\|_D$。整理上式得到

$$\|\Phi w^*-v_\pi\|_D\leq \frac{1}{1-\gamma}\|Mv_\pi-v_\pi\|_D.$$

由于$\|Mv_\pi-v_\pi\|_D$是$v_\pi$到所有可能近似值空间的正交投影误差，因此它等于$v_\pi$与任意$\hat{v}(w)$之间误差的最小值：

$$\|Mv_\pi-v_\pi\|_D=\min_w\|\hat{v}(w)-v_\pi\|_D.$$

所以

$$\|\Phi w^*-v_\pi\|_D\leq \frac{1}{1-\gamma}\min_w\|\hat{v}(w)-v_\pi\|_D.\tag{8.32}$$

最后补充证明上面用到的两个事实。

**1. 证明$\|M\|_D=1$。**

由加权范数定义，

$$\|x\|_D=\sqrt{x^TDx}=\|D^{1/2}x\|_2.$$

对应的诱导矩阵范数为

$$\|A\|_D=\max_{x\neq0}\frac{\|Ax\|_D}{\|x\|_D}=\|D^{1/2}AD^{-1/2}\|_2.$$

因此

$$\|M\|_D=\|\Phi(\Phi^TD\Phi)^{-1}\Phi^TD\|_D
=\|D^{1/2}\Phi(\Phi^TD\Phi)^{-1}\Phi^TDD^{-1/2}\|_2=1,$$

因为最后这个矩阵是一个正交投影矩阵，而正交投影矩阵的$L_2$范数为$1$。

**2. 证明对任意$x\in\mathbb{R}^n$有$\|P_\pi x\|_D\leq\|x\|_D$。**

首先

$$\|P_\pi x\|_D^2=x^TP_\pi^TDP_\pi x.$$

将其展开并重组可得

$$\begin{aligned}
\|P_\pi x\|_D^2
&=\sum_k [D]_{kk}\left(\sum_i[P_\pi]_{ki}x_i\right)^2\\
&\leq \sum_k [D]_{kk}\left(\sum_i[P_\pi]_{ki}x_i^2\right)\\
&=\sum_i\left(\sum_k[D]_{kk}[P_\pi]_{ki}\right)x_i^2\\
&=\sum_i[D]_{ii}x_i^2\\
&=\|x\|_D^2.
\end{aligned}$$

其中不等式使用了Jensen不等式，倒数第二个等号使用了$d_\pi^TP_\pi=d_\pi^T$。

---
