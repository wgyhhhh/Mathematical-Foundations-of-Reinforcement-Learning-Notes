---
title: 代码实战-🚀🚀 练习两小时半，从零复现PPO算法
date: 2026-03-30 18:02:18
categories:
  - 代码实战
tags:
    - LLM
    - RLHF
toc: true
cover: /imgs/pratical-3/1.png
---

# 简介

PPO (Proximal Policy Optimization, 近端策略优化)作为OpenAI于2017年提出的一种强化学习算法，作为RL4LLM领域适用性最广的算法之一，理解PPO算法的原理并从0实现对于初学者来说至关重要，因此本文致力于讲解PPO算法的原理并同时从零复现复现PPO算法。

简单来讲，PPO算法的核心在于四个模型和两个损失，四个模型分别为

- 策略模型：待优化的模型，参与参数更新
- 价值模型：计算期望回报，参与参数更新
- 参考模型：由策略模型初始化而来，用以计算KL散度，防止其偏离，不参与参数更新
- 奖励模型：计算当前动作的即时奖励，不参与参数更新

两个损失为策略损失(用于优化策略模型)与价值损失(用于优化价值模型)。

# 2. 基础概念

## 2.1 术语

- **策略**：在PPO中，策略即策略模型，即我们将要优化的大模型，拥有参数$\theta$，记为$\pi_\theta$。
  
    我们可以计算某个轨迹$\tau$发生的概率为：
      $$
      \begin{aligned}
      \pi_\theta(\tau) &= p(s_1) \pi_\theta(a_1 | s_1) p(s_2 | s_1, a_1) \pi_\theta(a_2 | s_2) p(s_3 | s_2, a_2) \cdots \\
      &= p(s_1) \prod_{t=1}^{T-1} \pi_\theta(a_t | s_t) p(s_{t+1} | s_t, a_t), \tag{1}
      \end{aligned}
      $$
其中$p(s_{t+1}|s_t,a_t)$由环境产生，与策略模型无关。

- **轨迹**：轨迹由一系列状态$s_t$、动作$a_t$组成，代表一次完整的采样，即大模型生成一个完整的句子。
    $$\tau=(s_0,a_0,s_1,a_1,...,s_{T-1},a_{T-1},s_T)$$
- **即时奖励(单步奖励)**: $r_t$奖励函数时状态-动作对的函数，记为$r_t=R(s_t,a_t)$。输入 $s_1, a_1$，会输出$r_1$，以此类推。
- **累计奖励**: 一条完整轨迹的所有奖励相加，就得到了$R(\tau)=\sum r_t$

## 2.2 基于策略的强化学习的优化目标

优化目标为优化策略模型$\pi_\theta$，以最大化期望奖励：

$$\arg \max_\theta J(\theta) = \arg \max_\theta \mathbb{E}_{\tau \sim \pi_\theta(\tau)}[R(\tau)]= \arg \max_\theta \sum_\tau R(\tau) \pi_\theta(\tau)$$

首先计算梯度，对$\theta$进行求导，即可表示为$\nabla_\theta$，这里及后续所有梯度，都是对$\theta$求导，已知奖励函数$R(\tau)$与$\theta$无关。则有：

$$\nabla J(\theta) = \sum_\tau R(\tau) \nabla \pi_\theta(\tau)$$

!!! note
    对于对数函数$f(x)=log\;x$，有:
    $$\nabla f(x)=f(x)\nabla \log f(x)$$

由上面结论可以得到$\nabla \pi_\theta(\tau) = \pi_\theta(\tau) \nabla \log \pi_\theta(\tau)$，代入上面的目标函数梯度：

$$
\begin{aligned}
\nabla J(\theta) &= \sum_\tau R(\tau) \nabla \pi_\theta(\tau) \\
&= \sum_\tau R(\tau) \pi_\theta(\tau) \nabla \log \pi_\theta(\tau), \tag{2}
\end{aligned}
$$

上式也可以写为期望形式：
$$\nabla J(\theta) =\mathbb{E}_{\tau \sim \pi_\theta(\tau)}[ R(\tau)  \nabla \log \pi_\theta(\tau)]$$

基于式(1)，$\nabla \pi_\theta(\tau)$的具体计算过程可写为
$$
\begin{aligned}
\nabla \log \pi_\theta(\tau) &= \nabla \log \left( p(s_1) \prod_{t=1}^{T} \pi_\theta(a_t | s_t) p(s_{t+1} | s_t, a_t) \right) \\
&= \nabla \left( \log p(s_1) + \sum_{t=1}^{T} \log \pi_\theta(a_t | s_t) + \sum_{t=1}^{T} \log p(s_{t+1} | s_t, a_t) \right) \\
&= \nabla \log p(s_1) + \nabla \sum_{t=1}^{T} \log \pi_\theta(a_t | s_t) + \nabla \sum_{t=1}^{T} \log p(s_{t+1} | s_t, a_t)
\end{aligned}
$$

其中$p(s_1)$与$p(s_{t+1}|s_t,a_t)$与策略模型参数$\theta$无关的梯度为0，可舍去。于是：

$$
\begin{aligned}
\nabla \log \pi_\theta(\tau) &= \nabla \sum_{t=1}^{T} \log \pi_\theta(a_t | s_t) \\
&= \sum_{t=1}^{T} \nabla \log \pi_\theta(a_t | s_t), \tag{3}
\end{aligned}
$$

最终得到：

$$
\begin{aligned}
\nabla J(\theta) &= \mathbb{E}_{\tau \sim \pi_\theta(\tau)}[ R(\tau) \nabla \log \pi_\theta(\tau)] \\
&= \mathbb{E}_{\tau \sim \pi_\theta(\tau)}[R(\tau)\sum_{t=1}^{T} \nabla \log \pi_\theta(a_t | s_t)]
\end{aligned}
$$

但是上述所有内容考虑的情况仅仅是完整轨迹，用整条轨迹的奖励去评估单步价值不是特别合适，单步奖励高不代表轨迹中每一步奖励都高。因此，有许多方法在$R(\tau)$上进行改进：

$$\nabla J(\theta) =\mathbb{E}_{\tau \sim \pi_\theta(\tau)}[\Psi(\tau)\sum_{t=1}^{T} \nabla \log \pi_\theta(a_t | s_t)]$$

其中$\Psi(\tau)$有以下若干种实现形式：

- **折扣奖励**: 折扣奖励可以平衡即时奖励和长期奖励之间的关系，可以决定模型是否短视，
    $$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-t-1} r_{T-1}
    = r_t + \gamma G_{t+1},\tag{4}$$
    其中$r_t$是$(s_t,a_t)$的函数，记为$r_t=R(s_t,a_t)$，$\gamma$为折扣因子，$\gamma$越小，模型越短视，详细证明可见[链接](https://wgyhhh.top/Mathematical-Foundations-of-Reinforcement-Learning-Notes/Chapter-3/3-5/#_1)。

- **状态价值函数**：对于式(4)，同一个状态$s$与给定的策略$\pi_\theta$，其回报$G_t$可能有所不同，可见[链接](https://wgyhhh.top/Mathematical-Foundations-of-Reinforcement-Learning-Notes/Chapter-2/2-3/)，而我们想要知道从某个状态出发，遵循某个策略平均可以拿到多少回报而不可考虑特定的动作$a_t$。
    $$V(s_t)=\mathbb{E}_{\pi}[G_t|s_t]$$

- **动作价值函数**：在给定策略和动作下，衡量某个动作的长期价值。
    $$Q(s_t,a_t)=\mathbb{E}_\pi[G_t|s_t,a_t]$$

    关于动作价值函数的详细介绍，可见[链接](https://wgyhhh.top/Mathematical-Foundations-of-Reinforcement-Learning-Notes/Chapter-2/2-8/)。
- **引入基线**：对于式(3)，其中奖励$R(\tau)$永远为整数，那一个$(s,a)$对的奖励值很小，其采样概率也会上升，而没有被采样到的$(s,a)$对，即便其奖励再大，其采样概率也会下降。因此，引入基线$b$，即可通过$R(\tau)-b>0$，则让$(s,a)$采样概率上升，反之下降。

- **优势函数**：
    $$A(s_t,a_t)=Q(s_t,a_t)-V(s_t)$$
- **时序差分算法**：
    $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

# 3. PPO算法

## 3.1 策略损失

由上我们可以知道对于动作价值函数为

$$
\begin{aligned}
A(s_t,a_t) &= Q(s_t,a_t)-V(s_t) \\
&= \mathbb{E}_{s_{t+1}\sim P(\cdot\mid s_t,a_t)}[r+\gamma V_\pi(s_{t+1})]
\end{aligned}
$$

推导过程可见[链接](https://wgyhhh.top/Mathematical-Foundations-of-Reinforcement-Learning-Notes/Chapter-2/2-8/)。

则优势函数为

$$
\begin{aligned}
A(s_t,a_t) &= Q(s_t,a_t)-V_\pi(s_t) \\
&= \mathbb{E}_{s_{t+1}\sim P(\cdot\mid s_t,a_t)}[r_t+\gamma V_\pi(s_{t+1})-V_\pi(s_t)]
\end{aligned}
$$

如果$V_\pi$可以准确衡量出策略$\pi$的值，那么$r_t+\gamma V_\pi(s_{t+1})-V_\pi(s_t)$项(也称为TD误差项)即为优势函数的无偏估计，但是该值往往通过神经网络预测得到而存在偏差问题，因此引入广义优势估计GAE，综合未来多步的TD误差，并引入一个0到1之间的新参数$\lambda$来进行加权平均：

$$
\begin{aligned}
\hat{A}_t &= \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + (\gamma \lambda)^3 \delta_{t+3} + \cdots \\
&= \sum_{l=0}^{\infty} (\gamma \lambda)^l(r_t+\gamma V_\pi(s_{t+1})-V_\pi(s_t))
\end{aligned}
$$

当$\lambda=0$时，仅考虑单步TD误差，对应于低方差，高偏差，当$\lambda=1$时，考虑未来所有步的TD误差，高方差，低偏差。

!!! note
    当神经网络初始未收敛时，$V_\pi$预测偏差大，那么通过$\lambda=1$更多采用即时奖励来降低偏差，但同时引入了更多的随机变量，因此带来了高方差。

因此GAE核心思想就是通过调节超参数来平衡方差和偏差。

根据式$\nabla J(\theta) =\mathbb{E}_{\tau \sim \pi_\theta(\tau)}[\Psi(\tau)\sum_{t=1}^{T} \nabla \log \pi_\theta(a_t | s_t)]$可知，梯度是根据当前策略$\pi_\theta$生成的数据来计算的。这意味着每次更新$\theta$后，$\pi_\theta$都会改变，因此每批数据只能用一次就作废。因此引入**重要性采样**，核心思想利用旧策略$\pi_{\theta_{\text{old}}}$生成的数据来估计新策略$\pi_\theta$的梯度。

$$
\begin{aligned}
\nabla J(\theta) &=
{E}_{\tau \sim \pi_\theta(\tau)}\left[\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}\Psi_t\sum_{t=1}^{T} \nabla \log \pi_\theta(a_t | s_t)\right] \\
&= {E}_{\tau \sim \pi_\theta(\tau)}\left[\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}\Psi_t\frac{\nabla\pi_\theta(a_t | s_t)}{\pi_\theta(a_t | s_t)}\right] \\
&= {E}_{\tau \sim \pi_\theta(\tau)}\left[\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}\Psi_t\right]
\end{aligned}
$$

此时轨迹是从旧策略模型采样得到的，通过一个比例调节因子，即可用于优化当前模型：

$$\argmax_{\pi_\theta}J(\pi_\theta)=\argmax_{\pi_\theta}{E}_{\tau \sim \pi_\theta(\tau)}[\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}\Psi_t]$$

若新旧策略差异较大，会导致训练不稳定，在这里PPO算法有两种变体。

1. clip裁剪

    通过对$\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$进行裁剪，将其限制在特定的范围内，避免更新幅度过大，从而得到了策略损失的最终形式。

    $$\argmax_{\pi_\theta}J(\pi_\theta)=\argmax_{\pi_\theta}{\mathbb{E}}_{\tau \sim \pi_\theta(\tau)}[\min\left(\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}\Psi_t,\text{clip}\left(\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)},1-\epsilon,1+\epsilon\right)\Psi_t\right)]$$

2. 加入KL散度

$$\arg \max_{\pi_\theta} J(\pi_\theta) = \arg \max_{\pi_\theta} {\mathbb{E}}_{\tau \sim \pi_{\theta^\text{old}}} \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta^\text{old}}(a_t | s_t)} \Psi_t - \beta \text{KL}(\pi_{\theta^\text{old}}(\cdot | s_t), \pi_\theta(\cdot | s_t)) \right]$$

## 3.2 价值损失

价值损失目标是尽可能准确估计状态$s_t$的期望价值$V_\phi({s_t})$，其目标函数定义为与状态期望价值的MSE，通过梯度下降发使其最小化：

$$\argmin_\phi J_V(\varphi) =\argmin_\phi \mathbb{E}_t \left[ \left( V(s_t) - V_\phi({s_t}) \right)^2 \right]$$

也可以使用时序差分目标，为

$$V_\phi({s_t})=r_t+\gamma V(s_{t+1})$$

如果使用GAE目标，则为

$$V_\phi({s_t})=\hat{A}_t+V(s_t)$$

当然这里也可以加入裁剪策略。

# 4. 从零复现

```bash
OpenRLHF
 ├── openrlhf
 │   ├── cli							// 训练入口函数
 │   │   ├── ...
 │   │   └── train_ppo.py
 │   ├── datasets						// 数据集处理相关
 │   ├── models						// 定义模型、loss相关
 │   │   ├── __init__.py
 │   │   ├── actor.py					// 定义 actor model 
 │   │   ├── loss.py					// 定义不同的 loss 函数
 │   │   ├── model.py					// 定义 critic model 和 reward model
 │   │   └── utils.py
 │   ├── trainer								
 │   │   └── ppo_trainer.py			// 定义 ppo 训练方法
 │   └── utils							// 工具类、函数定义
 │   │   └── remote_rm_utils.py 		// 可以自定义 reward model，通过 http 部署
 └── examples							// 示例启动脚本
     └── scripts
		 ├── ...
         └── train_ppo_llama.sh
```

在复现一章，我们将使用[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/tree/main)框架进行PPO有关训练脚本、损失函数和模型架构的讲解。

## 4.1 训练Reward Model

```
{
    "column_names": [
        "rejected",
        "rejected_score",
        "chosen_score",
        "chosen"
    ],
    "first_sample": {
        "rejected": [
            {
                "content": "Part 1. Definition\nYou are given a math word problem and you are supposed to apply division mathematical operator on the numbers embedded in the text to answer the following question and then only report the final numerical answer.\nPart 2. Example\nContext: Alex wants to split a collection of eggs into 4 groups. Alex has 8 eggs and 6 marbles. \n\n Question: How many eggs will each group have?\nAnswer: 2\nExplanation: Total eggs each group will get is 8/4 = 2\nPart 3. Exercise\nContext: 6 packs of dvds can be bought with 120 dollars. \n\n Question: How much does each pack cost?\nAnswer:",
                "role": "user"
            },
            {
                "content": "Your task is to define the mathematical operation to be performed on the given numbers to solve the problem and provide a formula to calculate the answer. In this case, the task involves dividing the total amount of eggs by the number of groups, and your answer will be the number of eggs each group will have. I will provide you with the formula to solve the exercise, and if you have any questions, feel free to ask.",
                "role": "assistant"
            }
        ],
        "rejected_score": 1.5,
        "chosen_score": 3.75,
        "chosen": [
            {
                "content": "Part 1. Definition\nYou are given a math word problem and you are supposed to apply division mathematical operator on the numbers embedded in the text to answer the following question and then only report the final numerical answer.\nPart 2. Example\nContext: Alex wants to split a collection of eggs into 4 groups. Alex has 8 eggs and 6 marbles. \n\n Question: How many eggs will each group have?\nAnswer: 2\nExplanation: Total eggs each group will get is 8/4 = 2\nPart 3. Exercise\nContext: 6 packs of dvds can be bought with 120 dollars. \n\n Question: How much does each pack cost?\nAnswer:",
                "role": "user"
            },
            {
                "content": "6 packs x $120/pack = $720\nExplanation: To find how much each pack costs, we need to divide the total cost by the number of packs. 120 dollars can buy 6 packs, so each pack costs $120/6 = $20.",
                "role": "assistant"
            }
        ]
    }
}
```

可以发现训练数据为`rejected`和`chosen`表示偏好，并且给出了`rejected_score`和`chosen_score`作为label。

在`openrlhf/cli/train_rm.py`中15行至155行进行`args`的解析后，就进入了`train`函数。

```python
def train(args):
    # DataSpeed初始化配置
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # 加载预训练模型
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        attn_implementation=args.attn_implementation,
        param_dtype=args.param_dtype,  # default: bf16
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        init_value_head=True,
        value_head_prefix=args.value_head_prefix,
        packing_samples=args.packing_samples,
    )

    # 加载tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    strategy.print(model)

    # 加载优化器
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    # 准备数据
    train_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.dataset_split,
    )

    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    train_dataset = RewardDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
    )

    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )

    if getattr(args, "eval_dataset", None):
        eval_data = blending_datasets(
            args.eval_dataset,
            None,  # No probability sampling for eval datasets
            strategy,
            dataset_split=args.eval_split,
        )
    else:
        # Used for calculating mean/std for reward normalization
        eval_data = train_data.select(range(min(args.max_samples, int(len(train_data) * 0.01))))

    eval_dataset = RewardDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,
        False,
        eval_dataset.collate_fn,
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # 多卡分配
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = RewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        max_epochs=args.max_epochs,
        loss=args.loss,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )
    # 开始训练
    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # Save value_head_prefix
    strategy.print("Save value_head_prefix in config")
    unwrap_model = strategy._unwrap_model(model)
    unwrap_model.config.value_head_prefix = args.value_head_prefix

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.save_path)
```

加下来，我们对其中的各个关键函数取出进行讲解：

### 4.1.1 从本地加载模型

```python
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    param_dtype="bf16",
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    attn_implementation="flash_attention_2",
    ds_config: dict = None,
    init_value_head=False,
    value_head_prefix="score",
    device_map=None,
    packing_samples=False,
    **kwargs,
) -> nn.Module:
    
    assert (
            model_type == "critic" or model_type == "reward"
        ), f"invalid model_type: {model_type}, should be critic or reward."
        # 从预训练模型中加载config配置，比如模型词表大小，Top-K参数。
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config.normalize_reward = normalize_reward
        config._attn_implementation = attn_implementation
    
        # 得到模型最后的value head
        value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
        logger.info(f"set value_head_prefix to `{value_head_prefix}`")
        # 加载预训练模型
        base_class = AutoModel._model_mapping[type(config)]
        base_pretrained_class = base_class.__base__
        # 在模型基础上加载自定义头
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
    
        # Note: dschf is defined in function scope to avoid global effects
        # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
    
        # Determine torch dtype based on param_dtype parameter, default: bf16
        from openrlhf.utils.utils import convert_to_torch_dtype
    
        torch_dtype = convert_to_torch_dtype(param_dtype)
    
        if load_in_4bit:
            assert param_dtype == "bf16", "we only support bnb_4bit_compute_dtype = bf16"
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            nf4_config = None
        # 加载预训练模型参数
        model = cls_class.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,  # default: bf16
            quantization_config=nf4_config,
            device_map=device_map,
            **kwargs,
        )
    
        # Lora参数配置
        if lora_rank > 0:
            model.enable_input_require_grads()
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
    
            if load_in_4bit:
                for name, module in model.named_modules():
                    if isinstance(module, LoraLayer):
                        module = module.to(torch.bfloat16)
                    if "norm" in name:
                        module = module.to(torch.float32)
                    if value_head_prefix in name or "embed_tokens" in name:
                        if hasattr(module, "weight"):
                            module = module.to(torch.bfloat16)
    
        # MoE - balancing loss
        model_config = model.config.to_dict()
        if "output_router_logits" in model_config:
            print("[MoE] set output_router_logits as True")
            model.config.output_router_logits = True
    
        set_z3_leaf_modules(model)
    
        # https://github.com/huggingface/transformers/issues/26877
        model.config.use_cache = False
    
        # NOTE: For reward model training only, intialize value_head manually
        # because deepspeed.zero.Init() will not intialize them.
        # TODO: Find a better way to clarify reward model training.
        if init_value_head:
            value_head = getattr(model, value_head_prefix)
            if dschf is not None:
                logger.info("initialize value_head for ZeRO-3 reward model training.")
                with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
            else:
                value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
    
        return model
```

看奖励模型内部结构：

```python
def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            # 设置基础模型及配置
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            # 设置value head的输入维度为预训练模型hidden size，输出维度为1
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # 设置归一化参数
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

            # Required by Transformers v5 to register derived model metadata such as
            # all_tied_weights_keys before from_pretrained() finalizes loading.
            self.post_init()
            # 之后还有前向传播，等训练时讲解
```

### 4.1.2 加载数据集

加载完模型后，继续加载tokenizer和optimizer

```python
tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

strategy.print(model)

# configure optimizer
optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)
```

然后加载数据集，重点管理数据处理过程

```python
train_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.dataset_split,
    )

train_data = train_data.select(range(min(args.max_samples, len(train_data))))
train_dataset = RewardDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
    )

# prepare dataloader
train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )

if getattr(args, "eval_dataset", None):
    eval_data = blending_datasets(
            args.eval_dataset,
            None,  # No probability sampling for eval datasets
            strategy,
            dataset_split=args.eval_split,
        )
else:
        # Used for calculating mean/std for reward normalization
    eval_data = train_data.select(range(min(args.max_samples, int(len(train_data) * 0.01))))

eval_dataset = RewardDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        input_template=args.input_template,
    )
eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,
        False,
        eval_dataset.collate_fn,
    )
```


首先`blending_datasets`混合数据集，通过`args.max_samples`限制最大样本数量，从而防止内存或计算资源溢出。

然后创建训练数据集和验证数据集`RewardDataset`。

```python
class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.extras = processed_dataset["extra"]

    def process_data(self, data):
        prompt, chosen, reject, margin = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.apply_chat_template,
            self.is_dpo,
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "extra": prompt_ids_len if self.is_dpo else margin,
        }
```

`process_data`函数的目标就是从原始数据中提取出来数据并用字典表示。




