---
title: BOX7.4
comments: true  # 开启评论
---
给定策略 $\pi$，其行动值可通过Expected Sarsa算法进行评估，该算法是Sarsa的一种变体。Expected Sarsa算法的表达式为

$$\begin{aligned}
q_{t+1}(s_t,a_t)
&=q_t(s_t,a_t)-\alpha_t(s_t,a_t)\left[q_t(s_t,a_t)-\left(r_{t+1}+\gamma\mathbb{E}[q_t(s_{t+1},A)]\right)\right],\\
q_{t+1}(s,a)&=q_t(s,a),\quad \text{for all }(s,a)\neq(s_t,a_t),
\end{aligned}$$

其中

$$\mathbb{E}[q_t(s_{t+1},A)]=\sum_a\pi_t(a|s_{t+1})q_t(s_{t+1},a)\doteq v_t(s_{t+1})$$

是在策略$\pi_t$下$q_t(s_{t+1},a)$的期望值。

Expected Sarsa与Sarsa的表达式非常相似，二者只是在TD目标上不同。Expected Sarsa的TD目标是

$$r_{t+1}+\gamma\mathbb{E}[q_t(s_{t+1},A)],$$

而Sarsa的TD目标是

$$r_{t+1}+\gamma q_t(s_{t+1},a_{t+1}).$$

由于该算法中包含期望值，因此被称为Expected Sarsa。尽管计算期望值会稍微增加计算复杂度，但它有一个好处：可以降低估计方差。原因是Sarsa中的随机变量为

$$\{s_t,a_t,r_{t+1},s_{t+1},a_{t+1}\},$$

而Expected Sarsa中的随机变量减少为

$$\{s_t,a_t,r_{t+1},s_{t+1}\}.$$

与式$(7.1)$中的TD学习算法类似，Expected Sarsa也可以看作是用于求解下面方程的随机近似算法：

$$q_\pi(s,a)=\mathbb{E}\left[R_{t+1}+\gamma\mathbb{E}[q_\pi(S_{t+1},A_{t+1})|S_{t+1}]\,\middle|\,S_t=s,A_t=a\right],\quad \forall s,a.\tag{7.15}$$

上式一开始看起来可能有些奇怪。事实上，它是贝尔曼方程的另一种表达。将

$$\mathbb{E}[q_\pi(S_{t+1},A_{t+1})|S_{t+1}]
=\sum_{A'}q_\pi(S_{t+1},A')\pi(A'|S_{t+1})
=v_\pi(S_{t+1})$$

代入式$(7.15)$，可得

$$q_\pi(s,a)=\mathbb{E}\left[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s,A_t=a\right],$$

这显然就是贝尔曼方程。

Expected Sarsa的实现方式与Sarsa类似，更多细节可参见文献[3,36,37]。

---
