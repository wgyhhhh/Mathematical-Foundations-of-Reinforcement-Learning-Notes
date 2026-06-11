---
title: BOX3.1
comments: true  # 开启评论
---
本节证明压缩映射定理。证明分为四个部分：先证明由$x_{k}=f(x_{k-1})$产生的序列收敛，再证明其极限点是固定点，随后证明固定点唯一，最后证明收敛速度是指数级的。

**第一步：证明序列$\{x_k\}_{k=1}^{\infty}$收敛。**

证明将用到柯西序列(Cauchy sequence)的概念。对于序列$x_1,x_2,\ldots\in\mathbb{R}$，如果对任意足够小的$\varepsilon>0$，都存在一个$N$，使得对所有$m,n>N$都有

$$\|x_m-x_n\|<\varepsilon,$$

那么称该序列为柯西序列。直观来说，这意味着从某一个有限位置$N$之后，序列中的所有元素彼此都足够接近。柯西序列的重要性在于：柯西序列一定收敛到某个极限点。

需要注意的是，柯西序列要求对所有$m,n>N$都有$\|x_m-x_n\|<\varepsilon$。如果仅有$x_{n+1}-x_n\to 0$，并不足以说明该序列是柯西序列。例如，当$x_n=\sqrt{n}$时有$x_{n+1}-x_n\to0$，但显然$x_n=\sqrt{n}$是发散的。

下面证明$\{x_k=f(x_{k-1})\}_{k=1}^{\infty}$是柯西序列，因此它收敛。首先，因为$f$是压缩映射，所以有

$$\|x_{k+1}-x_k\|=\|f(x_k)-f(x_{k-1})\|\leq\gamma\|x_k-x_{k-1}\|.$$

同理可得$\|x_k-x_{k-1}\|\leq\gamma\|x_{k-1}-x_{k-2}\|,\ldots,\|x_2-x_1\|\leq\gamma\|x_1-x_0\|$。因此

$$\begin{aligned}
\|x_{k+1}-x_k\|
&\leq \gamma\|x_k-x_{k-1}\|\\
&\leq \gamma^2\|x_{k-1}-x_{k-2}\|\\
&\leq \cdots\\
&\leq \gamma^k\|x_1-x_0\|.
\end{aligned}$$

由于$\gamma<1$，给定任意$x_1,x_0$，$\|x_{k+1}-x_k\|$都会随着$k\to\infty$以指数速度收敛到零。不过，$\{\|x_{k+1}-x_k\|\}$收敛并不足以推出$\{x_k\}$收敛。因此我们需要进一步考察任意$m>n$时的$\|x_m-x_n\|$：

$$\begin{aligned}
\|x_m-x_n\|
&=\|x_m-x_{m-1}+x_{m-1}-\cdots-x_{n+1}+x_{n+1}-x_n\|\\
&\leq \|x_m-x_{m-1}\|+\cdots+\|x_{n+1}-x_n\|\\
&\leq \gamma^{m-1}\|x_1-x_0\|+\cdots+\gamma^n\|x_1-x_0\|\\
&=\gamma^n(\gamma^{m-1-n}+\cdots+1)\|x_1-x_0\|\\
&\leq \gamma^n(1+\cdots+\gamma^{m-1-n}+\gamma^{m-n}+\gamma^{m-n+1}+\cdots)\|x_1-x_0\|\\
&=\frac{\gamma^n}{1-\gamma}\|x_1-x_0\|.
\end{aligned}\tag{3.4}$$

因此，对任意$\varepsilon$，总能找到一个$N$，使得对所有$m,n>N$都有$\|x_m-x_n\|<\varepsilon$。所以该序列是柯西序列，从而收敛到某个极限点，记为

$$x^*=\lim_{k\to\infty}x_k.$$

**第二步：证明极限点$x^*$是固定点。**

由于

$$\|f(x_k)-x_k\|=\|x_{k+1}-x_k\|\leq\gamma^k\|x_1-x_0\|,$$

可知$\|f(x_k)-x_k\|$以指数速度收敛到零。因此在极限处有

$$f(x^*)=x^*.$$

也就是说，$x^*$是$f$的固定点。

**第三步：证明固定点唯一。**

假设还存在另一个固定点$x'$，满足$f(x')=x'$。那么

$$\|x'-x^*\|=\|f(x')-f(x^*)\|\leq\gamma\|x'-x^*\|.$$

由于$\gamma<1$，上式成立当且仅当$\|x'-x^*\|=0$。因此$x'=x^*$，固定点是唯一的。

**第四步：证明$x_k$以指数速度收敛到$x^*$。**

回顾式$(3.4)$中已经证明

$$\|x_m-x_n\|\leq\frac{\gamma^n}{1-\gamma}\|x_1-x_0\|.$$

由于$m$可以任意大，因此有

$$\|x^*-x_n\|=\lim_{m\to\infty}\|x_m-x_n\|\leq\frac{\gamma^n}{1-\gamma}\|x_1-x_0\|.$$

由于$\gamma<1$，当$n\to\infty$时，误差以指数速度收敛到零。

---
