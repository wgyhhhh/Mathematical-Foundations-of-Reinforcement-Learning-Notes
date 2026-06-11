---
title: BOX4.1
comments: true  # 开启评论
---
由于$v_{\pi_{k+1}}$和$v_{\pi_k}$都是状态值，它们分别满足贝尔曼方程：

$$v_{\pi_{k+1}}=r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_{\pi_{k+1}},$$

$$v_{\pi_k}=r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}.$$

又因为

$$\pi_{k+1}=\arg\max_{\pi}(r_\pi+\gamma P_\pi v_{\pi_k}),$$

所以

$$r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_{\pi_k}\geq r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k}.$$

因此

$$\begin{aligned}
v_{\pi_k}-v_{\pi_{k+1}}
&=(r_{\pi_k}+\gamma P_{\pi_k}v_{\pi_k})-(r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_{\pi_{k+1}})\\
&\leq (r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_{\pi_k})-(r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_{\pi_{k+1}})\\
&=\gamma P_{\pi_{k+1}}(v_{\pi_k}-v_{\pi_{k+1}}).
\end{aligned}$$

于是

$$v_{\pi_k}-v_{\pi_{k+1}}\leq \gamma^2P_{\pi_{k+1}}^2(v_{\pi_k}-v_{\pi_{k+1}})\leq \cdots \leq \gamma^nP_{\pi_{k+1}}^n(v_{\pi_k}-v_{\pi_{k+1}}).$$

令$n\to\infty$，由于$\gamma^n\to0$，且$P_{\pi_{k+1}}^n$对任意$n$都是非负随机矩阵，因此

$$v_{\pi_k}-v_{\pi_{k+1}}\leq \lim_{n\to\infty}\gamma^nP_{\pi_{k+1}}^n(v_{\pi_k}-v_{\pi_{k+1}})=0.$$

所以

$$v_{\pi_{k+1}}\geq v_{\pi_k}.$$

这证明了策略改进步骤确实不会让策略变差。

---
