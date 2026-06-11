---
title: BOX9.5
comments: true  # 开启评论
---
证明分为三步。

**第一步：证明$v_\pi^*$是泊松方程的一个解。**

为简化记号，令

$$A\doteq I_n-P_\pi+\mathbf{1}_n d_\pi^T.$$

则$v_\pi^*=A^{-1}r_\pi$。$A$可逆性将在第三步证明。将$v_\pi^*=A^{-1}r_\pi$代入式$(9.23)$，需要验证

$$A^{-1}r_\pi=r_\pi-\mathbf{1}_nd_\pi^Tr_\pi+P_\pi A^{-1}r_\pi.$$

该式等价于

$$(-A^{-1}+I_n-\mathbf{1}_nd_\pi^T+P_\pi A^{-1})r_\pi=0,$$

也等价于

$$(-I_n+A-\mathbf{1}_nd_\pi^TA+P_\pi)A^{-1}r_\pi=0.$$

括号中的项为零，因为

$$\begin{aligned}
&-I_n+A-\mathbf{1}_nd_\pi^TA+P_\pi\\
&=-I_n+(I_n-P_\pi+\mathbf{1}_nd_\pi^T)
-\mathbf{1}_nd_\pi^T(I_n-P_\pi+\mathbf{1}_nd_\pi^T)+P_\pi\\
&=0.
\end{aligned}$$

因此$v_\pi^*$是泊松方程的一个解。

**第二步：给出解的一般形式。**

将$\bar{r}_\pi=d_\pi^Tr_\pi$代入式$(9.23)$，可得

$$v_\pi=r_\pi-\mathbf{1}_nd_\pi^Tr_\pi+P_\pi v_\pi,\tag{9.25}$$

因此

$$ (I_n-P_\pi)v_\pi=(I_n-\mathbf{1}_nd_\pi^T)r_\pi.\tag{9.26}$$

注意$I_n-P_\pi$是奇异矩阵，因为对任意$\pi$都有

$$ (I_n-P_\pi)\mathbf{1}_n=0.$$

所以式$(9.26)$的解并不唯一。如果$v_\pi^*$是一个解，那么对任意$x\in\mathrm{Null}(I_n-P_\pi)$，$v_\pi^*+x$也是一个解。当$P_\pi$不可约时，

$$\mathrm{Null}(I_n-P_\pi)=\mathrm{span}\{\mathbf{1}_n\}.$$

因此泊松方程的任意解都可以写为

$$v_\pi^*+c\mathbf{1}_n,\quad c\in\mathbb{R}.$$

**第三步：证明$A=I_n-P_\pi+\mathbf{1}_nd_\pi^T$可逆。**

下面使用一个引理：

!!! info
    **引理9.3**. 矩阵$I_n-P_\pi+\mathbf{1}_nd_\pi^T$可逆，且其逆矩阵可表示为

    $$[I_n-(P_\pi-\mathbf{1}_nd_\pi^T)]^{-1}
    =\sum_{k=1}^{\infty}(P_\pi^k-\mathbf{1}_nd_\pi^T)+I_n.$$

证明引理。首先使用两个事实：若矩阵$M$的谱半径$\rho(M)<1$，则$I-M$可逆；并且$\rho(M)<1$当且仅当$\lim_{k\to\infty}M^k=0$。

因此只需证明

$$\lim_{k\to\infty}(P_\pi-\mathbf{1}_nd_\pi^T)^k=0.$$

注意

$$ (P_\pi-\mathbf{1}_nd_\pi^T)^k=P_\pi^k-\mathbf{1}_nd_\pi^T,\quad k\geq1.\tag{9.27}$$

该式可用归纳法证明。例如$k=2$时，

$$\begin{aligned}
(P_\pi-\mathbf{1}_nd_\pi^T)^2
&=P_\pi^2-P_\pi\mathbf{1}_nd_\pi^T-\mathbf{1}_nd_\pi^TP_\pi+\mathbf{1}_nd_\pi^T\mathbf{1}_nd_\pi^T\\
&=P_\pi^2-\mathbf{1}_nd_\pi^T,
\end{aligned}$$

其中用到了

$$P_\pi\mathbf{1}_n=\mathbf{1}_n,\qquad d_\pi^TP_\pi=d_\pi^T,\qquad d_\pi^T\mathbf{1}_n=1.$$

更高阶情形类似。

由于$d_\pi$是稳态分布，

$$\lim_{k\to\infty}P_\pi^k=\mathbf{1}_nd_\pi^T.$$

由式$(9.27)$可知

$$\lim_{k\to\infty}(P_\pi-\mathbf{1}_nd_\pi^T)^k=0.$$

因此$\rho(P_\pi-\mathbf{1}_nd_\pi^T)<1$，从而$I_n-(P_\pi-\mathbf{1}_nd_\pi^T)$可逆。其逆矩阵为

$$\begin{aligned}
(I_n-(P_\pi-\mathbf{1}_nd_\pi^T))^{-1}
&=\sum_{k=0}^{\infty}(P_\pi-\mathbf{1}_nd_\pi^T)^k\\
&=I_n+\sum_{k=1}^{\infty}(P_\pi^k-\mathbf{1}_nd_\pi^T).
\end{aligned}$$

由于

$$I_n-(P_\pi-\mathbf{1}_nd_\pi^T)=I_n-P_\pi+\mathbf{1}_nd_\pi^T=A,$$

所以$A$可逆。证明完毕。

---
