---
title: BOX9.3
comments: true  # 开启评论
---
由于$d_0(s)$与策略$\pi$无关，因此

$$\nabla_\theta\bar{v}_{\pi,0}
=\nabla_\theta\sum_{s\in\mathcal{S}}d_0(s)v_\pi(s)
=\sum_{s\in\mathcal{S}}d_0(s)\nabla_\theta v_\pi(s).$$

将引理9.2中给出的$\nabla_\theta v_\pi(s)$表达式代入上式，可得

$$\begin{aligned}
\nabla_\theta\bar{v}_{\pi,0}
&=\sum_{s\in\mathcal{S}}d_0(s)\sum_{s'\in\mathcal{S}}\Pr_\pi(s'|s)
\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s',\theta)q_\pi(s',a)\\
&=\sum_{s'\in\mathcal{S}}\left[\sum_{s\in\mathcal{S}}d_0(s)\Pr_\pi(s'|s)\right]
\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s',\theta)q_\pi(s',a)\\
&\doteq \sum_{s'\in\mathcal{S}}\rho_\pi(s')\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s',\theta)q_\pi(s',a)\\
&=\sum_{s\in\mathcal{S}}\rho_\pi(s)\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)\\
&=\sum_{s\in\mathcal{S}}\rho_\pi(s)\sum_{a\in\mathcal{A}}\pi(a|s,\theta)\nabla_\theta\ln\pi(a|s,\theta)q_\pi(s,a)\\
&=\mathbb{E}[\nabla_\theta\ln\pi(A|S,\theta)q_\pi(S,A)],
\end{aligned}$$

其中$S\sim\rho_\pi$，$A\sim\pi(S,\theta)$。证明完毕。

---
