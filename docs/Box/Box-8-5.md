---
title: BOX8.5
comments: true  # 开启评论
---
下面证明

$$w^*=A^{-1}b$$

是最小化投影贝尔曼误差$J_{PBE}(w)$的最优解。由于

$$J_{PBE}(w)=0\Longleftrightarrow \hat{v}(w)-MT_\pi(\hat{v}(w))=0,$$

因此只需研究方程

$$\hat{v}(w)=MT_\pi(\hat{v}(w))$$

的根。

在线性情形下，将$\hat{v}(w)=\Phi w$和$M$的表达式代入上式，可得

$$\Phi w=\Phi(\Phi^TD\Phi)^{-1}\Phi^TD(r_\pi+\gamma P_\pi\Phi w).\tag{8.31}$$

由于$\Phi$列满秩，因此对任意$x,y$有

$$\Phi x=\Phi y\Longleftrightarrow x=y.$$

所以式$(8.31)$推出

$$\begin{aligned}
w&=(\Phi^TD\Phi)^{-1}\Phi^TD(r_\pi+\gamma P_\pi\Phi w)\\
&\Longleftrightarrow \Phi^TD(r_\pi+\gamma P_\pi\Phi w)=(\Phi^TD\Phi)w\\
&\Longleftrightarrow \Phi^TDr_\pi+\gamma\Phi^TDP_\pi\Phi w=(\Phi^TD\Phi)w\\
&\Longleftrightarrow \Phi^TDr_\pi=\Phi^TD(I-\gamma P_\pi)\Phi w\\
&\Longleftrightarrow w=(\Phi^TD(I-\gamma P_\pi)\Phi)^{-1}\Phi^TDr_\pi\\
&=A^{-1}b.
\end{aligned}$$

其中$A$和$b$由式$(8.21)$给出。因此

$$w^*=A^{-1}b$$

就是最小化$J_{PBE}(w)$的最优解。

---
