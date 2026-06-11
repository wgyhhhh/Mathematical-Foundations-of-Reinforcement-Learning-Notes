---
title: BOX9.4
comments: true  # 开启评论
---
由$\bar{v}_\pi$的定义，

$$\begin{aligned}
\nabla_\theta\bar{v}_\pi
&=\nabla_\theta\sum_{s\in\mathcal{S}}d_\pi(s)v_\pi(s)\\
&=\sum_{s\in\mathcal{S}}\nabla_\theta d_\pi(s)v_\pi(s)
+\sum_{s\in\mathcal{S}}d_\pi(s)\nabla_\theta v_\pi(s).
\end{aligned}\tag{9.20}$$

上式包含两项。先处理第二项。将式$(9.17)$中$\nabla_\theta v_\pi$的表达式代入，可得

$$\begin{aligned}
\sum_{s\in\mathcal{S}}d_\pi(s)\nabla_\theta v_\pi(s)
&=(d_\pi^T\otimes I_m)\nabla_\theta v_\pi\\
&=(d_\pi^T\otimes I_m)[(I_n-\gamma P_\pi)^{-1}\otimes I_m]u\\
&=[d_\pi^T(I_n-\gamma P_\pi)^{-1}\otimes I_m]u.
\end{aligned}\tag{9.21}$$

注意

$$d_\pi^T(I_n-\gamma P_\pi)^{-1}=\frac{1}{1-\gamma}d_\pi^T,$$

这一点可通过在等式两边右乘$(I_n-\gamma P_\pi)$验证。因此式$(9.21)$变为

$$\sum_{s\in\mathcal{S}}d_\pi(s)\nabla_\theta v_\pi(s)
=\frac{1}{1-\gamma}\sum_{s\in\mathcal{S}}d_\pi(s)\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a).$$

另一方面，式$(9.20)$中的第一项包含$\nabla_\theta d_\pi$。但是由于第二项包含因子$1/(1-\gamma)$，当$\gamma\to1$时第二项占主导，而第一项可以忽略。因此

$$\nabla_\theta\bar{v}_\pi
\approx \frac{1}{1-\gamma}\sum_{s\in\mathcal{S}}d_\pi(s)\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a).$$

进一步，由$\bar{r}_\pi=(1-\gamma)\bar{v}_\pi$可得

$$\begin{aligned}
\nabla_\theta\bar{r}_\pi
&=(1-\gamma)\nabla_\theta\bar{v}_\pi\\
&\approx \sum_{s\in\mathcal{S}}d_\pi(s)\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)\\
&=\sum_{s\in\mathcal{S}}d_\pi(s)\sum_{a\in\mathcal{A}}\pi(a|s,\theta)\nabla_\theta\ln\pi(a|s,\theta)q_\pi(s,a)\\
&=\mathbb{E}[\nabla_\theta\ln\pi(A|S,\theta)q_\pi(S,A)].
\end{aligned}$$

上面的近似要求当$\gamma\to1$时，式$(9.20)$中的第一项不会趋于无穷。更多讨论可参见文献[66, Section 4]。

---
