---
title: BOX7.3
comments: true  # 开启评论
---
如第$2.8.2$节所介绍，基于行动值表示的贝尔曼方程为

$$\begin{aligned}
q_\pi(s,a)
&=\sum_r rp(r|s,a)
+\gamma\sum_{s'}\sum_{a'}q_\pi(s',a')p(s'|s,a)\pi(a'|s')\\
&=\sum_r rp(r|s,a)
+\gamma\sum_{s'}p(s'|s,a)\sum_{a'}q_\pi(s',a')\pi(a'|s').
\end{aligned}\tag{7.14}$$

该方程刻画了不同行动值之间的关系。由于

$$\begin{aligned}
p(s',a'|s,a)
&=p(s'|s,a)p(a'|s',s,a)\\
&=p(s'|s,a)p(a'|s')\\
&=p(s'|s,a)\pi(a'|s'),
\end{aligned}$$

其中第二个等号来自条件独立性，所以式$(7.14)$可改写为

$$q_\pi(s,a)=\sum_r rp(r|s,a)+\gamma\sum_{s'}\sum_{a'}q_\pi(s',a')p(s',a'|s,a).$$

根据期望的定义，上式等价于式$(7.13)$。因此，式$(7.13)$就是用行动值表示的贝尔曼方程。

---
