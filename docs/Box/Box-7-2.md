---
title: BOX7.2
comments: true  # 开启评论
---
我们基于第6章的定理6.3证明TD学习的收敛性。为此，首先需要构造一个与定理6.3形式一致的随机过程。

考虑任意状态$s\in\mathcal{S}$。在时刻$t$，由式$(7.1)$中的TD算法可知，如果$s=s_t$，则

$$v_{t+1}(s)=v_t(s)-\alpha_t(s)\left(v_t(s)-(r_{t+1}+\gamma v_t(s_{t+1}))\right).\tag{7.7}$$

如果$s\neq s_t$，则

$$v_{t+1}(s)=v_t(s).\tag{7.8}$$

定义估计误差为

$$\Delta_t(s)\doteq v_t(s)-v_\pi(s),$$

其中$v_\pi(s)$是在策略$\pi$下状态$s$的真实状态值。对式$(7.7)$两边同时减去$v_\pi(s)$，可得

$$\begin{aligned}
\Delta_{t+1}(s)
&=(1-\alpha_t(s))\Delta_t(s)
+\alpha_t(s)\underbrace{(r_{t+1}+\gamma v_t(s_{t+1})-v_\pi(s))}_{\eta_t(s)}\\
&=(1-\alpha_t(s))\Delta_t(s)+\alpha_t(s)\eta_t(s),\quad s=s_t.
\end{aligned}\tag{7.9}$$

对式$(7.8)$两边同时减去$v_\pi(s)$，可得

$$\Delta_{t+1}(s)=\Delta_t(s)=(1-\alpha_t(s))\Delta_t(s)+\alpha_t(s)\eta_t(s),\quad s\neq s_t,$$

其中此时$\alpha_t(s)=0$且$\eta_t(s)=0$。因此，无论$s$是否等于$s_t$，都有统一表达式

$$\Delta_{t+1}(s)=(1-\alpha_t(s))\Delta_t(s)+\alpha_t(s)\eta_t(s).$$

这正是定理6.3中的随机过程形式。下面证明定理6.3中的三个条件成立。

第一个条件由定理7.1的假设直接给出。接下来证明第二个条件，即对所有$s\in\mathcal{S}$有

$$\|\mathbb{E}[\eta_t(s)|\mathcal{H}_t]\|_\infty\leq \gamma\|\Delta_t(s)\|_\infty.$$

其中$\mathcal{H}_t$表示历史信息。由于马尔科夫性质，当给定$s$时，$\eta_t(s)=r_{t+1}+\gamma v_t(s_{t+1})-v_\pi(s)$或$\eta_t(s)=0$不再依赖历史信息。因此

$$\mathbb{E}[\eta_t(s)|\mathcal{H}_t]=\mathbb{E}[\eta_t(s)].$$

当$s\neq s_t$时，$\eta_t(s)=0$，因此

$$|\mathbb{E}[\eta_t(s)]|=0\leq \gamma\|\Delta_t(s)\|_\infty.\tag{7.10}$$

当$s=s_t$时，有

$$\begin{aligned}
\mathbb{E}[\eta_t(s)]
&=\mathbb{E}[\eta_t(s_t)]\\
&=\mathbb{E}[r_{t+1}+\gamma v_t(s_{t+1})-v_\pi(s_t)|s_t]\\
&=\mathbb{E}[r_{t+1}+\gamma v_t(s_{t+1})|s_t]-v_\pi(s_t).
\end{aligned}$$

又因为

$$v_\pi(s_t)=\mathbb{E}[r_{t+1}+\gamma v_\pi(s_{t+1})|s_t],$$

所以上式变为

$$\begin{aligned}
\mathbb{E}[\eta_t(s)]
&=\gamma\mathbb{E}[v_t(s_{t+1})-v_\pi(s_{t+1})|s_t]\\
&=\gamma\sum_{s'\in\mathcal{S}}p(s'|s_t)[v_t(s')-v_\pi(s')].
\end{aligned}$$

因此

$$\begin{aligned}
|\mathbb{E}[\eta_t(s)]|
&=\gamma\left|\sum_{s'\in\mathcal{S}}p(s'|s_t)[v_t(s')-v_\pi(s')]\right|\\
&\leq \gamma\sum_{s'\in\mathcal{S}}p(s'|s_t)\max_{s'\in\mathcal{S}}|v_t(s')-v_\pi(s')|\\
&=\gamma\max_{s'\in\mathcal{S}}|v_t(s')-v_\pi(s')|\\
&=\gamma\|\Delta_t(s)\|_\infty.
\end{aligned}\tag{7.11}$$

由式$(7.10)$和$(7.11)$可知，对所有$s\in\mathcal{S}$都有

$$\|\mathbb{E}[\eta_t(s)]\|_\infty\leq \gamma\|\Delta_t(s)\|_\infty,$$

这就是定理6.3中的第二个条件。

最后看第三个条件。当$s=s_t$时，

$$\mathrm{var}[\eta_t(s)|\mathcal{H}_t]
=\mathrm{var}[r_{t+1}+\gamma v_t(s_{t+1})-v_\pi(s_t)|s_t]
=\mathrm{var}[r_{t+1}+\gamma v_t(s_{t+1})|s_t].$$

当$s\neq s_t$时，

$$\mathrm{var}[\eta_t(s)|\mathcal{H}_t]=0.$$

由于$r_{t+1}$是有界的，所以第三个条件也可以被证明成立。由定理6.3可知，$\Delta_t(s)\to0$几乎必然成立，因此TD学习算法收敛。

---
