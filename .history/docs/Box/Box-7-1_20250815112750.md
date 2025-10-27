接下来我们证明，$(7.1)$中的TD算法可通过应用RM算法求解$(7.4)$获得。

对于状态$s_t$，我们定义一个函数为

$g(v_\pi(s_t))= v_\pi(s_t)-\mathbb{E}\left[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s_t\right].$

那么，$(7.4)$式等价于

$$g(v_\pi(s_t))=0.$$

我们的目标是通过RM算法求解上述方程以获得$v_\pi(s_t)$。由于可以获得$r_{t+1}$和$s_{t+1}$（它们分别是$R_{t+1}$和$S_{t+1}$的样本），因此可获得的$g(v_\pi(s_t))$噪声观测值为

$$\begin{aligned}\tilde{g}(v_{\pi}(s_{t}))&=v_{\pi}(s_{t})-\begin{bmatrix}r_{t+1}+\gamma v_{\pi}(s_{t+1})\end{bmatrix}\\&=\underbrace{\left(v_{\pi}(s_{t})-\mathbb{E}\left[R_{t+1}+\gamma v_{\pi}(S_{t+1})|S_{t}=s_{t}\right]\right)}_{g(v_{\pi}(s_{t}))}\\&+\underbrace{\left(\mathbb{E}\left[R_{t+1}+\gamma v_{\pi}(S_{t+1})|S_{t}=s_{t}\right]-\left[r_{t+1}+\gamma v_{\pi}(s_{t+1})\right]\right).}_{\eta}\end{aligned}$$

因此，用于求解$g(v_\pi(s_t)) =0$的 RM算法(第$6.2$节)

$$\begin{aligned}v_{t+1}(s_{t})&=v_{t}(s_{t})-\alpha_{t}(s_{t})\tilde{g}(v_{t}(s_{t}))\\&=v_{t}(s_{t})-\alpha_{t}(s_{t})\left(v_{t}(s_{t})-\left[r_{t+1}+\gamma v_{\pi}(s_{t+1})\right]\right)\end{aligned},\tag{7.5}$$

其中$v_t(s_t)$表示时刻$t$时$v_\pi(s_t)$的估计值，$\alpha_t(s_t)$为学习率。

!!! note 
    7.1中的式子为$v_{t+1}(s_t) = v_t(s_t) - \alpha_t(s_t) \left[ v_t(s_t) - \left( r_{t+1} + \gamma v_t(s_{t+1}) \right) \right]$

$(7.5)$式中的算法与$(7.1)$式TD算法具有相似的表达式，唯一区别在于$(7.5)$式右侧包含$v_\pi(s_{t+1})$，而$(7.1)$式包含$v_t(s_{t+1})$。这是因为$(7.5)$式在设计时仅需估计状态$s_t$的价值，并假设其他状态价值已知。若需估计所有状态的价值，则应将右侧的$v_\pi(s_{t+1})$替换为$v_t(s_{t+1})$。此时，$(7.5)$式将与$(7.1)$式完全等同。但这样的替换是否仍能保证收敛性？答案是肯定的，这将在定理$7.1$中予以证明。