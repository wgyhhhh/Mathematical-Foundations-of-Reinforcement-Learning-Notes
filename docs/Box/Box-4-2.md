---
title: BOX4.2
comments: true  # 开启评论
---
证明思路是说明：策略迭代算法的收敛速度不慢于值迭代算法。

为证明$\{v_{\pi_k}\}_{k=0}^{\infty}$收敛，引入另一个由值迭代算法生成的序列$\{v_k\}_{k=0}^{\infty}$：

$$v_{k+1}=f(v_k)=\max_\pi(r_\pi+\gamma P_\pi v_k).$$

前面已经知道，对任意初始值$v_0$，值迭代序列$v_k$都会收敛到最优状态值$v^*$。

对于$k=0$，总可以找到一个$v_0$，使得对任意$\pi_0$都有$v_{\pi_0}\geq v_0$。下面用归纳法证明

$$v_k\leq v_{\pi_k}\leq v^*,\quad \forall k.$$

假设对某个$k\geq0$有$v_{\pi_k}\geq v_k$。考察$k+1$：

$$\begin{aligned}
v_{\pi_{k+1}}-v_{k+1}
&=(r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_{\pi_{k+1}})-\max_\pi(r_\pi+\gamma P_\pi v_k)\\
&\geq (r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_{\pi_k})-\max_\pi(r_\pi+\gamma P_\pi v_k)\\
&=(r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_{\pi_k})-(r_{\pi'_k}+\gamma P_{\pi'_k}v_k)\\
&\geq (r_{\pi'_k}+\gamma P_{\pi'_k}v_{\pi_k})-(r_{\pi'_k}+\gamma P_{\pi'_k}v_k)\\
&=\gamma P_{\pi'_k}(v_{\pi_k}-v_k),
\end{aligned}$$

其中

$$\pi'_k=\arg\max_\pi(r_\pi+\gamma P_\pi v_k).$$

上面第一个不等式使用了Box 4.1中的结论$v_{\pi_{k+1}}\geq v_{\pi_k}$，第二个不等式使用了策略改进步骤

$$\pi_{k+1}=\arg\max_\pi(r_\pi+\gamma P_\pi v_{\pi_k}).$$

由于归纳假设给出$v_{\pi_k}-v_k\geq0$，且$P_{\pi'_k}$为非负矩阵，因此

$$v_{\pi_{k+1}}-v_{k+1}\geq0.$$

所以$v_{\pi_{k+1}}\geq v_{k+1}$。由归纳法可得，对所有$k$都有$v_k\leq v_{\pi_k}$。

另一方面，任意策略的状态值都不超过最优状态值，因此$v_{\pi_k}\leq v^*$。综上，

$$v_k\leq v_{\pi_k}\leq v^*.$$

由于值迭代中的$v_k\to v^*$，根据夹逼思想可知$v_{\pi_k}\to v^*$。因此，策略迭代生成的状态值序列收敛到最优状态值，对应的策略序列也收敛到一个最优策略。

!!! note
    这里的核心是构造一个值迭代序列$v_k$作为“下界”。既然$v_k$会收敛到$v^*$，而策略迭代的$v_{\pi_k}$始终夹在$v_k$和$v^*$之间，那么$v_{\pi_k}$也只能收敛到$v^*$。

---
