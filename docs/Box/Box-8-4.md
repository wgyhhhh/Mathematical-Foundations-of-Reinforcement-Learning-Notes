---
title: BOX8.4
comments: true  # 开启评论
---
矩阵$A$是正定的，指的是对任意合适维度的非零向量$x$都有

$$x^TAx>0.$$

若矩阵正定，则它一定可逆。下面证明

$$A=\Phi^TD(I-\gamma P_\pi)\Phi$$

是正定的。

证明思路是先证明

$$M\doteq D(I-\gamma P_\pi)\succ0.\tag{8.28}$$

如果$M\succ0$，并且$\Phi$是列满秩矩阵（假设特征向量线性无关），则

$$A=\Phi^TM\Phi\succ0.$$

注意

$$M=\frac{M+M^T}{2}+\frac{M-M^T}{2}.$$

由于$M-M^T$是斜对称矩阵，因此对任意$x$都有

$$x^T(M-M^T)x=0.$$

所以$M\succ0$当且仅当$M+M^T\succ0$。为证明$M+M^T\succ0$，使用严格对角占优矩阵正定这一事实。

首先有

$$ (M+M^T)\mathbf{1}_n>0.\tag{8.29}$$

证明如下。由于$P_\pi\mathbf{1}_n=\mathbf{1}_n$，

$$M\mathbf{1}_n=D(I-\gamma P_\pi)\mathbf{1}_n=D(\mathbf{1}_n-\gamma\mathbf{1}_n)=(1-\gamma)d_\pi.$$

另一方面，

$$M^T\mathbf{1}_n=(I-\gamma P_\pi^T)D\mathbf{1}_n=(I-\gamma P_\pi^T)d_\pi=(1-\gamma)d_\pi,$$

其中最后一个等号使用了$P_\pi^Td_\pi=d_\pi$。因此

$$ (M+M^T)\mathbf{1}_n=2(1-\gamma)d_\pi.$$

由于$d_\pi$的所有元素均为正，所以式$(8.29)$成立。

式$(8.29)$的逐元素形式为

$$\sum_{j=1}^{n}[M+M^T]_{ij}>0,\quad i=1,\ldots,n.$$

也就是

$$[M+M^T]_{ii}+\sum_{j\neq i}[M+M^T]_{ij}>0.$$

根据$M=D(I-\gamma P_\pi)$的表达式可知，$M$的对角元素为正，非对角元素非正。因此上式可写为

$$[M+M^T]_{ii}>\sum_{j\neq i}|[M+M^T]_{ij}|.$$

这说明$M+M^T$的每一行中，对角元素的绝对值都大于同一行其他非对角元素绝对值之和。因此$M+M^T$是严格对角占优矩阵，从而正定。

于是$M\succ0$，进而

$$A=\Phi^TM\Phi\succ0,$$

所以$A$可逆。

---
