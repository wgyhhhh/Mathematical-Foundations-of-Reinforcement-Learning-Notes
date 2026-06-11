---
title: BOX8.3
comments: true  # 开启评论
---
利用全期望公式，有

$$\begin{aligned}
&\mathbb{E}\left[\left(r_{t+1}+\gamma\phi^T(s_{t+1})w_t-\phi^T(s_t)w_t\right)\phi(s_t)\right]\\
&=\sum_{s\in\mathcal{S}}d_\pi(s)\mathbb{E}\left[r_{t+1}\phi(s_t)|s_t=s\right]\\
&\quad+\sum_{s\in\mathcal{S}}d_\pi(s)\mathbb{E}\left[\phi(s_t)\left(\gamma\phi^T(s_{t+1})-\phi^T(s_t)\right)w_t|s_t=s\right].
\end{aligned}\tag{8.24}$$

这里假设$s_t$服从稳态分布$d_\pi$。

先看式$(8.24)$中的第一项。注意

$$\mathbb{E}[r_{t+1}\phi(s_t)|s_t=s]=\phi(s)\mathbb{E}[r_{t+1}|s_t=s]=\phi(s)r_\pi(s),$$

其中

$$r_\pi(s)=\sum_a\pi(a|s)\sum_r rp(r|s,a).$$

因此第一项可写为

$$\sum_{s\in\mathcal{S}}d_\pi(s)\phi(s)r_\pi(s)=\Phi^TDr_\pi.\tag{8.25}$$

再看第二项。由于

$$\begin{aligned}
&\mathbb{E}\left[\phi(s_t)(\gamma\phi^T(s_{t+1})-\phi^T(s_t))w_t|s_t=s\right]\\
&=-\phi(s)\phi^T(s)w_t+\gamma\phi(s)\mathbb{E}[\phi^T(s_{t+1})|s_t=s]w_t\\
&=-\phi(s)\phi^T(s)w_t+\gamma\phi(s)\sum_{s'\in\mathcal{S}}p(s'|s)\phi^T(s')w_t,
\end{aligned}$$

所以第二项变为

$$\begin{aligned}
&\sum_{s\in\mathcal{S}}d_\pi(s)\mathbb{E}\left[\phi(s_t)(\gamma\phi^T(s_{t+1})-\phi^T(s_t))w_t|s_t=s\right]\\
&=\sum_{s\in\mathcal{S}}d_\pi(s)\phi(s)\left[-\phi(s)+\gamma\sum_{s'\in\mathcal{S}}p(s'|s)\phi(s')\right]^Tw_t\\
&=\Phi^TD(-\Phi+\gamma P_\pi\Phi)w_t\\
&=-\Phi^TD(I-\gamma P_\pi)\Phi w_t.
\end{aligned}\tag{8.26}$$

结合式$(8.25)$和式$(8.26)$，可得

$$\mathbb{E}\left[\left(r_{t+1}+\gamma\phi^T(s_{t+1})w_t-\phi^T(s_t)w_t\right)\phi(s_t)\right]
=\Phi^TDr_\pi-\Phi^TD(I-\gamma P_\pi)\Phi w_t.$$

令

$$b\doteq\Phi^TDr_\pi,\qquad A\doteq\Phi^TD(I-\gamma P_\pi)\Phi,$$

于是期望可写为

$$b-Aw_t.$$

---
