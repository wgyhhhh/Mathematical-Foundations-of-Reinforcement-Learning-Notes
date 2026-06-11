---
title: BOX3.2
comments: true  # 开启评论
---
考虑任意两个向量$v_1,v_2\in\mathbb{R}^{|\mathcal{S}|}$，并令

$$\pi_1^* \doteq \arg\max_{\pi}(r_\pi+\gamma P_\pi v_1),\qquad
\pi_2^* \doteq \arg\max_{\pi}(r_\pi+\gamma P_\pi v_2).$$

那么

$$\begin{aligned}
f(v_1)&=\max_\pi(r_\pi+\gamma P_\pi v_1)=r_{\pi_1^*}+\gamma P_{\pi_1^*}v_1\geq r_{\pi_2^*}+\gamma P_{\pi_2^*}v_1,\\
f(v_2)&=\max_\pi(r_\pi+\gamma P_\pi v_2)=r_{\pi_2^*}+\gamma P_{\pi_2^*}v_2\geq r_{\pi_1^*}+\gamma P_{\pi_1^*}v_2,
\end{aligned}$$

其中$\geq$表示逐元素比较。因此

$$\begin{aligned}
f(v_1)-f(v_2)
&=r_{\pi_1^*}+\gamma P_{\pi_1^*}v_1-(r_{\pi_2^*}+\gamma P_{\pi_2^*}v_2)\\
&\leq r_{\pi_1^*}+\gamma P_{\pi_1^*}v_1-(r_{\pi_1^*}+\gamma P_{\pi_1^*}v_2)\\
&=\gamma P_{\pi_1^*}(v_1-v_2).
\end{aligned}$$

类似地，也可以得到

$$f(v_2)-f(v_1)\leq \gamma P_{\pi_2^*}(v_2-v_1).$$

于是

$$\gamma P_{\pi_2^*}(v_1-v_2)\leq f(v_1)-f(v_2)\leq \gamma P_{\pi_1^*}(v_1-v_2).$$

定义

$$z\doteq \max\left\{|\gamma P_{\pi_2^*}(v_1-v_2)|,\ |\gamma P_{\pi_1^*}(v_1-v_2)|\right\}\in\mathbb{R}^{|\mathcal{S}|},$$

其中$\max(\cdot)$、$|\cdot|$和$\geq$均为逐元素运算。根据定义，$z\geq0$。一方面，不难看出

$$-z\leq \gamma P_{\pi_2^*}(v_1-v_2)\leq f(v_1)-f(v_2)\leq \gamma P_{\pi_1^*}(v_1-v_2)\leq z,$$

因此

$$|f(v_1)-f(v_2)|\leq z.$$

由此可得

$$\|f(v_1)-f(v_2)\|_\infty\leq \|z\|_\infty,\tag{3.5}$$

其中$\|\cdot\|_\infty$是最大范数。

另一方面，假设$z_i$是$z$的第$i$个元素，$p_i^T$和$q_i^T$分别是$P_{\pi_1^*}$和$P_{\pi_2^*}$的第$i$行。那么

$$z_i=\max\{\gamma|p_i^T(v_1-v_2)|,\gamma|q_i^T(v_1-v_2)|\}.$$

由于$p_i$的所有元素均非负，且元素之和为$1$，因此

$$|p_i^T(v_1-v_2)|\leq p_i^T|v_1-v_2|\leq \|v_1-v_2\|_\infty.$$

类似地，有

$$|q_i^T(v_1-v_2)|\leq \|v_1-v_2\|_\infty.$$

因此

$$z_i\leq \gamma\|v_1-v_2\|_\infty,$$

从而

$$\|z\|_\infty=\max_i |z_i|\leq \gamma\|v_1-v_2\|_\infty.$$

将该不等式代入$(3.5)$，可得

$$\|f(v_1)-f(v_2)\|_\infty\leq \gamma\|v_1-v_2\|_\infty.$$

这就证明了$f(v)$的压缩性质。

---
