---
title: BOX3.3
comments: true  # 开启评论
---
对任意策略$\pi$，都有

$$v_\pi=r_\pi+\gamma P_\pi v_\pi.$$

另一方面，由贝尔曼最优方程可知

$$v^*=\max_\pi(r_\pi+\gamma P_\pi v^*)=r_{\pi^*}+\gamma P_{\pi^*}v^*\geq r_\pi+\gamma P_\pi v^*.$$

因此

$$v^*-v_\pi\geq (r_\pi+\gamma P_\pi v^*)-(r_\pi+\gamma P_\pi v_\pi)=\gamma P_\pi(v^*-v_\pi).$$

反复应用上面的不等式，可得

$$v^*-v_\pi\geq \gamma P_\pi(v^*-v_\pi)\geq \gamma^2P_\pi^2(v^*-v_\pi)\geq\cdots\geq \gamma^nP_\pi^n(v^*-v_\pi).$$

于是

$$v^*-v_\pi\geq \lim_{n\to\infty}\gamma^nP_\pi^n(v^*-v_\pi)=0.$$

最后一个等号成立，是因为$\gamma<1$，且$P_\pi^n$是一个非负矩阵，其所有元素都不大于$1$（因为$P_\pi^n\mathbf{1}=\mathbf{1}$）。

因此，对任意策略$\pi$都有$v^*\geq v_\pi$。这说明贝尔曼最优方程的解$v^*$就是最优状态值。

---
