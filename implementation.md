# VLA-TTT Implementation Plan

更新日期：2026-11-17。当前状态：已实现模型、RoboMME-Plus HDF5 数据管线、训练与推理入口。已做基础工程检查，尚未加载完整模型、启动训练或模拟器。安装与命令见 [README.md](README.md)。

## 1. 目标与交付范围

在本仓库实现 **VLA-TTT**：以 **RoboMME-Plus** 为数据与环境接口，实现论文 *RoboTTT: Context Scaling for Robot Policies* 的模型机制、序列训练方法和有状态推理流程，提供视频条件模仿及带显式接管标签的 DAgger Distillation 接口。

仓库名称使用 `VLA-TTT`，Python 包使用 `vla_ttt`，模型类使用 `VLATTTPolicy`，命令行使用 `vla-ttt`。README 和论文引用保留原工作名称、作者及来源，说明本项目是独立复现。[R1][R2]

按用户要求，实施顺序改为：**直接接入 GR00T，加入 TTT 模块，完成数据管线、训练代码和流式推理，最后整理配置与使用文档。**

本次范围为代码复现，不安排独立的可微记忆正确性验证阶段，不开发 toy 模型、合成记忆任务或数值梯度验证工具，也不实施基线、消融、context scaling 扫描、仿真/真机评估或论文结果复现实验。

交付内容包括模型实现、三种数据模式、训练与恢复入口、有状态推理接口及配置文档。保留基础工程检查：导入、配置、张量形状、权重加载，以及环境允许时的前向/反向运行检查。提供训练能力，不将正式训练或实验结果作为本次交付条件。

## 2. 已核实的信息与第一步版本核对

### 2.1 论文明确的设计

以下作为实现约束，来源为论文 §2–3、附录 A。[R2]

| 项目 | 论文设置 |
| --- | --- |
| 基座 | GR00T N1.7；附录描述 VLM 为 Eagle |
| 动作头 | DiT；16 个层级加入 TTT |
| Fast model | 两层 MLP，GeLU 激活，梯度下降更新 |
| 内层目标 | key–value binding 的 MSE；先更新 W，再读取 Q |
| 外层目标 | 轨迹上的动作 flow-matching loss |
| Register tokens | 每个策略时刻 16 个，传递视觉语言信息 |
| 门控 | `attention_output + tanh(alpha) * ttt_output`；`alpha=0.001` 初始化 |
| 内层学习率 | 基础值 0.1，并学习其调节参数 |
| 位置编码 | RoPE，theta=10000 |
| 噪声采样 | 每个动作块独立采样 `tau=0.999*(1-u)`，`u~Beta(1.5,1)` |
| 跨时间状态 | 各层 fast weights；跨 TBPTT 段保留数值、截断梯度 |
| 推理 | slow weights 固定；每个 rollout 从学到的 W0 初始化 |

### 2.2 公开 GR00T 与论文并非已确认同一版本

截至核查日，官方 GR00T N1.7 代码和 `nvidia/GR00T-N1.7-3B` 权重可获取，但公开版本使用 **Cosmos-Reason2-2B/Qwen3-VL**，与论文附录的 Eagle 有差异。[R3][R4]

公开 checkpoint 包含 `num_layers=32`、`interleave_self_attention=true`。已按固定源码确认其为 16 对交替 cross/self blocks：全部保留，在每对之后插入一个 TTT 层。DiT 宽度 1536，动作 horizon 40，模型填充动作维度 132。[R3][R5][R6]

官方代码 commit 固定在安装脚本中，GR00T checkpoint/processor revision 和嵌套 Cosmos revision 固定在模型配置及 adapter 中。adapter 在加载时检查三个直接执行的关键上游源文件 SHA-256，并拒绝权重 missing/unexpected/mismatched keys。

`audit-backbone` 默认只输出预期来源，不加载权重。显式加 `--load-model` 后才构造模型并报告实际参数量、结构和统计文件 hash；该完整加载入口尚未在本机运行。

本次仅实现 `backbone_alignment=public_adaptation`。`paper_aligned` 当前会报错；未取得论文同版 Eagle 基座，不能把当前公开 checkpoint 当作同版权重。

当前 RoboTTT 官方主页未提供代码或权重入口；后续如发布官方实现，将核对差异并更新计划，不预设第三方实现等同于官方实现。[R1]

## 3. 技术栈与仓库结构

按用户选择，统一使用 **Python 3.12、PyTorch 2.9.1、torchvision 0.24.1**，Transformers 为 4.57.3。GR00T 与 RoboMME-Plus 在同一进程直接连接。GR00T 安装脚本仅覆盖其依赖元数据中的两个 Torch 版本，保留模型源码不变；这属于本项目的依赖覆盖，尚未经完整 GPU 运行确认。TTT 使用普通 PyTorch 算子直接集成。

本地执行静态代码与配置检查。完整基座的运行环境按 Linux + NVIDIA GPU 准备；实际训练显存取决于所选基座、batch 和 segment 配置，本文不承诺单卡能够运行 8K 训练。

已实现结构：

```text
VLA-TTT/
├── implementation.md
├── README.md
├── pyproject.toml
├── docs/{backbone,data}.md
├── configs/
│   ├── base.yaml
│   ├── model/groot_public.yaml
│   ├── train/{pretrain,posttrain,bc,video,dagger}.yaml
│   └── inference/default.yaml
├── src/vla_ttt/
│   ├── models/
│   │   ├── fast_weights.py       # 运行时状态、初始化、detach、reset
│   │   ├── ttt_mlp.py            # 可微 KVB 更新与读取
│   │   ├── ttt_block.py          # QKV、RoPE、门控、mask
│   │   ├── sequence_dit.py       # 逐时刻 attention + 跨时刻 TTT
│   │   ├── policy.py             # 统一训练与推理接口
│   │   └── adapters/             # 固定版本 GR00T
│   ├── data/
│   │   ├── schema.py
│   │   ├── episodes.py
│   │   ├── sampling.py
│   │   ├── collate.py
│   │   └── statistics.py         # 仅训练集归一化统计
│   ├── training/
│   │   ├── flow_matching.py
│   │   ├── tbptt.py
│   │   ├── trainer.py
│   │   └── checkpoint.py
│   ├── robomme/{observations,policy,runner}.py
│   └── cli.py
├── scripts/install_groot.sh
└── outputs/                     # 配置快照、checkpoint、运行日志
```

GR00T 的适配重点是 `Gr00tN1d7ActionHead`、DiT block 和 processor。尽量保留上游处理器、动作表示、归一化及参数名；通过明确的 adapter 接口加入时序状态，避免依赖难以追踪的运行时 hook。[R6][R7][R8]

## 4. 模型实现

### 4.1 单步信息与跨时间信息

对于第 i 次策略调用，原始输入为当前图像、共享语言指令、本体状态和带噪动作块：

```text
图像 + 指令 -> 固定版本 VLM -> VL tokens Phi_i
本体状态 -> state encoder -> q_i
带噪动作块 -> action encoder -> A_i_tokens
16 个 register tokens R_i

各 DiT 层级：
  当前时刻的 self/cross attention
  -> [R_i, q_i, A_i_tokens] 的 TTT update/apply
  -> tanh gate 与 attention 输出融合

action decoder -> flow velocity -> 去噪后的动作块
```

VL tokens 通过当前时刻的 attention 影响 register/state/action tokens，不直接把全部 VL tokens 送入 TTT。跨时间连接仅由 TTT 承担，避免在时间维上构造全注意力矩阵。[R2, §3.1]

### 4.2 可微 fast-weight 内循环

每层维护学习到的初始参数 `W0`，运行时为每个 batch 样本复制独立状态 `W[b, layer]`。实现采用显式函数式 MLP 张量，必要时使用 `functional_call`；运行状态不注册为全局模型参数。[R10]

```text
K_i, V_i, Q_i = learned_projections(X_i)
L_inner_i = masked_mean(|| fast_mlp(K_i; W_prev) - V_i ||²)
W_next = W_prev - eta_i * grad_W(L_inner_i)
O_i = fast_mlp(Q_i; W_next)
```

约束：

- 训练内循环保留求导图，例如 `autograd.grad(..., create_graph=True)`，使外层损失能优化 W0、K/V 投影和更新动态。[R9]
- 不通过 `.data`、原地覆盖或普通 optimizer step 实现 fast-weight 更新。
- 每个样本的 KVB loss 独立归一化；batch 向量化时避免把内层步长额外除以 batch size。
- 初始化状态连接到 W0 的计算图；TBPTT 边界才执行 detach。
- K/V 目标分支在训练时保留求导路径，不引入一阶近似。
- Fast weights、内层梯度和 reduction 默认使用 FP32；主干精度遵循所选基座配置。

已采用的工程细节：每层 `eta=0.1*softplus(log_lr)/log(2)`，`log_lr` 初始化为零；单个两层 MLP，含 bias，Xavier 权重初始化、零 bias；默认记忆宽度 1024、MLP hidden width 1536、输入 LayerNorm，门控为逐 hidden 维向量。维度与输入归一化可配置；bias、单头方式及初始化固定于实现。这些不能标作论文已公开参数。

### 4.3 时间因果性与更新 mini-batch

实现将同一次策略调用的 register/state/action tokens 作为一个 TTT 更新 mini-batch，按策略调用顺序 update-then-apply。论文图 2 提及 mini-batch，但具体 token 内更新方式未完整给出，此处标记为工程设定。

允许当前时刻 token 共同参与更新；禁止把未来时刻 token 合并进一次更新后用于预测过去。训练输入只使用合法的带噪动作块，干净动作作为 flow target；不会额外把当前干净目标直接写入记忆。

保留基座原有的 attention、归一化、残差连接与动作解码参数；新增 TTT 的输入输出维度与实际插入层匹配。

## 5. 数据协议与时间单位

### 5.1 统一 episode 数据

读取官方原版 `Yinpei/robomme_data_h5` 用于 pretrain，Plus `Yinpei/robomme_ood_data_h5` 用于 posttrain，不含 LeRobot 转换器。每条 episode 保存或派生以下信息：

| 字段 | 内容 |
| --- | --- |
| 标识 | episode_id、task_id、embodiment_id、源文件；保留 setup metadata |
| 时间 | control_hz、原始帧索引、控制时间跳跃标记 |
| 视觉语言 | 各相机 RGB、camera_valid、instruction |
| 状态动作 | proprio、executed_actions、各维有效 mask、动作坐标系/单位 |
| 行为来源 | robot / human_correction / demonstration / video_only |
| 上下文关系 | 同一 episode 的 demo 前缀与执行阶段、session reset 标记 |
| 处理信息 | 相机映射、归一化统计来源和 hash |

batch 的物理动作是 `actions[B,S,40,8]`，图像与命名状态通过 `LazyObservationGrid` 按 TBPTT 段读取为 `[B][segment_S]` 观测列表，再交给官方 processor。没有一次堆叠整条长序列图像。S 是记忆输入步数，Ha 是动作 horizon，二者均不直接等于论文的 8K。

输入使用 front/wrist RGB、7 个关节角及平均手指宽度、`task_goal`。动作使用绝对 `joint_action[8]`，夹爪 `-1=关闭,+1=打开`；不把 MoveCube 的 native15 截断成 8 维。GR00T 的 `new_embodiment` projector index 为 10。完整字段约定见 `docs/data.md`。

分别配置和记录：

- `raw_context_steps`：原始控制时间步数量及秒数。
- `observation_stride`：相邻策略输入对应的原始时间间隔。
- `action_horizon`：每次预测的动作数量。
- `execution_horizon`：每次实际执行的动作数量。
- `policy_steps`、`ttt_tokens`：策略调用次数与 TTT token 总量。

论文正文以 30Hz 原始时间步解释 8K，图 2 又以 `t=1,1+H` 展示动作块。采样 stride、H 和执行方式必须明确记录；不能把 8K tokens 或 8K 次策略调用直接作为论文 8K 的复现结果。[R2]

默认 `observation_stride=execution_horizon=action_horizon=40`，配置入口固定这套非重叠协议。Plus 记录频率是 20Hz；40 步对应 2 秒控制时间。`observation_stride` 在本实现中表示动作决策间隔：中间帧仍作为无动作监督的记忆输入，`demo_stride=1` 保留每帧演示。context 步使用零归一化动作、全零 action token mask 和 `tau=0`，训练与推理一致。此逐帧记忆策略是适配 RoboMME-Plus 的工程设定，不作为论文采样方案的已知事实。

### 5.2 独立的 mask

| 片段 | valid | state_update | action_loss |
| --- | ---: | ---: | ---: |
| 普通机器人示范决策帧 | 1 | 1 | 1 |
| 执行中的中间观测帧 | 1 | 1 | 0 |
| 人类视频上下文 | 1 | 1 | 0 |
| DAgger 机器人自主动作 | 1 | 1 | 0 |
| DAgger 人类纠正动作 | 1 | 1 | 1 |
| Padding | 0 | 0 | 0 |

`action_loss_mask` 支持动作块内部逐动作、逐维标注，防止接管发生在动作块中间时把机器人错误动作也作为纠正目标。padding 既不写入状态，也不改变有效位置计数。

loss=0 的上下文仍进行 TTT 更新；在同一 TBPTT 段内，后续监督损失可以通过它反传。配对视频到机器人执行之间保留记忆；无关 episode 之间重置。场景被重置不一定意味着上下文 session 应重置。

### 5.3 三条数据路径

1. **BC**：连续的机器人示范，监督有效动作。
2. **视频条件模仿**：同一配置的人类视频与机器人示范拼接；只有机器人部分监督。所有配置使用相同的任务指令，避免文字泄漏目标配置。缺失的 proprio/action 使用显式缺失标识和可配置占位编码，禁止生成伪动作标签；具体编码属于待核实设定。
3. **DAgger Distillation**：保留完整执行与接管历史，只监督人类纠正。存储真实执行动作、接管边界、行为策略版本，而非仅存策略预测动作。[R2, §3.3]

未接入 EgoScale。`pretrain` 使用原版 RoboMME 训练数据、冻结 GR00T 并训练新增参数；`posttrain` 默认加载该阶段 checkpoint，在 Plus 50 条训练轨迹上全参数适配。两阶段均使用 video 模式保留演示，并共用原版训练集拟合的归一化统计。原版默认 post_action 对齐、Plus 默认 pre_action，对齐设置及统计来源写入配置；实际原版 HDF5 的生成版本仍待核对。官方成功示范不当作接管数据，DAgger 要求额外来源标签。

数据加载器读取用户提供的 episode 清单，保留 configuration 和数据来源标识。仅在指定训练数据上拟合归一化统计。数据读取按 TBPTT 段解码和送入 GPU，避免一次保留全部长序列图像与 VLM 激活。

## 6. 训练流程

### 6.1 Sequence action forcing

对每个样本、每次策略调用独立采样噪声与 tau：

```text
u_i ~ Beta(1.5, 1)
tau_i = 0.999 * (1 - u_i)
epsilon_i ~ Normal(0, I)
A_noisy_i = tau_i * A_i + (1 - tau_i) * epsilon_i
target_velocity_i = A_i - epsilon_i
```

预测 velocity 的 MSE 按动作有效 mask 计算。每个序列先对有效监督动作维度和时间归一化，再对有监督序列求 batch 均值；分段 backward 必须使用整个逻辑序列预先计算的分母。该 masked reduction 是本项目明确采用的设定，论文未完全说明 masking 后的分母。

若整个逻辑 batch 都没有动作监督，数据校验应拒绝该训练样本组合，或显式跳过 outer optimizer 与 scheduler step；不能除以零，也不能把没有训练信号的 batch 计为有效训练步。

### 6.2 TBPTT 与 outer optimizer

采用以下参考策略：一个逻辑序列中的 slow weights 保持不变，每个 segment 单独 backward 累计梯度，全部 segment 完成后再执行一次 outer optimizer step。

```python
optimizer.zero_grad()
state = initialize_from_learned_W0(batch)
normalizer = compute_full_sequence_loss_normalizer(batch)

for segment in stream_segments(batch):
    prediction, next_state = model.forward_sequence(segment, state)
    loss = masked_flow_loss(prediction, segment, normalizer)
    if segment_has_supervision(segment):
        loss.backward()
    state = detach_fast_weights(next_state, requires_grad=True)

clip_outer_gradients_if_configured()
optimizer.step()
```

这是流程示意，正式实现需处理 batch 中不同样本的 reset、混合精度和分布式归一化。

重要细节：

- segment 边界保持 fast weights 数值，不重新初始化 W0，也不重置位置计数。
- 不在每个时间步 detach；否则外层损失无法学习段内的记忆更新。
- W0 仅从仍连接到它的第一段获得梯度。若第一段全是无监督视频，上述截断下 W0 可能没有任务梯度；必须记录这一情况，检查视频/监督边界的分段策略。
- 如果视频段在出现任何监督之前被 detach，该段参数也收不到后续动作损失的梯度。分段器优先移动边界，让相邻 context 与监督同段。单样本且段长至少 2 时可保护该局部连接；多样本边界冲突或段长为 1 时保留段长上限并警告。更早上下文仅保留状态数值，不能声称全视频都获得梯度。
- 冻结主干参数仍需允许梯度穿过插入层之间的冻结 DiT 运算。只对独立且冻结的 VLM 特征提取使用 no_grad。
- TBPTT 长度和 outer step 的具体频率未在论文完整公开；上述选择需记录为工程设定。
- 分布式阶段按全局有监督序列数归一化，显式 SUM 同步外层参数梯度，不对局部均值再求平均。无监督 rank 也参与必要同步；全局无监督 batch 跳过 optimizer/scheduler。
- 只同步外层模型参数梯度，不同步不同 episode 的运行状态。分布式入口采用可配置的数据并行；本次不新增 FSDP 或自定义高阶梯度通信实现。

### 6.3 论文训练配方

| 参数 | 预训练 | 下游任务训练 |
| --- | --- | --- |
| 更新范围 | 新增 TTT、投影、门控、register 等参数；基础 GR00T 冻结 | 全参数 |
| 步数 | 30,000 | 20,000 |
| 上下文 | 渐进增长至目标长度 | 1K 原始时间步 |
| 优化器 | AdamW | AdamW |
| 峰值 LR | 2e-5 | 5e-5 |
| Weight decay | 1e-5 | 1e-5 |
| Scheduler | Warmup-Stable-Decay | Cosine |
| 论文 GPU | 16 × GB200 | 8 × GB200 |
| 论文 global batch | ≤4K：64；8K：16 | 8 |

来源：论文 §3.4、附录 A.2。新增参数的分组清单由实现显式打印，避免漏掉 W0、学习率和 register tokens。[R2]

配置保留论文的阶段步数与 LR，但默认 batch=1、segment=8、完整 episode、空 curriculum；warmup/decay、梯度裁剪为工程默认值。视频前缀超过显式 context 预算时拒绝静默裁掉演示。`paper_aligned` 不可用；没有运行上述论文训练或扫描。

### 6.4 Checkpoint 与可恢复训练

保存 slow weights、W0、optimizer/scheduler、各 rank 随机数状态、数据位置、完整配置、统计文件 hash、数据顺序签名和上游 revisions。统计文件单独保存，恢复时核对 hash。只支持逻辑序列边界恢复，不保存段内 fast weights 或已累积梯度。

训练入口记录外层 loss、学习率、梯度范数、有监督序列数、step/epoch 等 JSONL 日志，checkpoint 保存完整配置和数据来源。没有单独记录每层 inner loss 或开展内层正确性实验。

## 7. 流式推理与状态生命周期

已实现接口：

```python
state = policy.init_state(batch_size, session_ids=session_ids)
state = policy.observe_context(video_batch, state)
actions, next_state = policy.act(observation, state)
state = next_state
state = policy.reset_state(state, reset_mask)
```

state 显式包含各层 fast weights、位置、session ID 和提交计数。模型共享 slow weights，每个机器人 session 独立持有 fast weights，避免跨任务或跨客户端串记忆。

推理阶段冻结所有 slow weights，TTT 内层局部启用梯度，更新后 detach。上游部分推理入口使用 no_grad，适配器需明确恢复内层求导；不能直接把整个有 TTT 的策略放进 inference_mode。[R7]

### 7.1 多步去噪的状态提交

论文没有完整说明同一观测的 k 次去噪前向如何更新持久记忆。本项目先采用以下可配置假设：

1. 进入当前策略调用时保存不可变 `W_prev`。
2. 每个去噪前向都从 `W_prev` 计算本次临时 `W_candidate`，只递推动作噪声状态。
3. 完成动作块生成后，默认提交最后一次前向产生的候选 fast weights，提交次数为一次。
4. `commit_rule` 写入 checkpoint 和运行日志，并在文档中说明其工程假设。

本次只实现上述提交规则，不新增其他提交方案及对照实验。持久状态的赋值放在去噪循环结束后，避免同一观测被累计提交 k 次；该方式标记为工程设定。

wrapper 缓存动作块，每次向环境返回一个 8 维动作；缓存期间收集全部新观测，在再次规划前按序写入。环境报告动作错误时丢弃剩余缓存并重新规划。单 episode runner 可通过回调记录原始交互，不额外生成纠正标签或执行反馈损失。

### 7.2 运行配置

控制器执行频率、策略调用频率、action horizon 和 execution horizon 分别配置。fast weights 作为固定形状状态保存；推理更新完成后释放旧计算图，不保存随历史增长的 KV cache。

## 8. 基础工程检查

本节仅检查实现能否接入和运行，不设独立算法验证阶段或实验指标：

- Python 模块可导入，依赖与配置项可解析。
- 基座权重可加载，新增参数与原有参数的键清晰区分。
- 数据字段、mask、张量形状、dtype 和 device 匹配。
- 在具备模型权重及 GPU 环境时，检查实际模型的前向、loss.backward 和单次推理入口能否执行。
- Checkpoint 保存/加载、session 初始化和 reset 接口可调用。

不以数值梯度对照、合成任务学习效果、成功率或基线差异作为交付门槛。

## 9. 实施里程碑

| 阶段 | 实现工作 | 阶段完成条件 |
| --- | --- | --- |
| P0 基座接入 | 固定版本、导出结构、复用 processor、加载 checkpoint | 明确实际基座及差异，建立 GR00T adapter |
| P1 TTT 集成 | FastWeightState、两层 MLP、QKV、门控、register tokens、reset | 在实际 DiT 插入 TTT，提供显式状态输入输出 |
| P2 数据管线 | episode schema、连续采样、三类 mask、BC/视频/DAgger 数据路径 | 数据可转换并组装为训练 batch |
| P3 训练代码 | Sequence action forcing、TBPTT、参数分组、scheduler、checkpoint | 提供预训练、任务训练和恢复入口，完成基础运行检查 |
| P4 流式推理 | session、上下文输入、多步去噪、单次状态提交 | 提供动作生成和状态生命周期接口 |
| P5 配置与文档 | 安装说明、启动命令、模型/数据/训练/推理配置 | 交付代码及使用说明，记录未公开细节的工程选择 |

上下文长度、TBPTT segment 和混合精度直接作为训练配置实现，不另设长上下文实验或性能优化阶段。

已提供以下命令行入口：

```bash
vla-ttt audit-backbone --config configs/model/groot_public.yaml
vla-ttt validate-data --data /path/to/plus_h5
vla-ttt prepare-data --data /path/to/plus_h5
vla-ttt train --config configs/train/pretrain.yaml
vla-ttt train --config configs/train/posttrain.yaml
vla-ttt infer --config configs/inference/default.yaml
```

## 10. 实现选择与未运行部分

以下区分已落地的工程选择与仍未验证的条件：

| 缺口 | 处理办法 |
| --- | --- |
| 论文同版基座及 checkpoint | 完成版本审计；公开基座适配配置保留差异标签 |
| 数据格式、动作表示及数据比例 | 原版/Plus HDF5、joint8、显式时间对齐、可选训练 manifest；无 LeRobot/EgoScale 转换器 |
| GPU、存储与运行环境 | 明确完整模型运行条件；缺少环境时如实记录未运行的入口 |
| MLP hidden width、bias、head、归一化及 W0 分布 | 1024→1536→1024、bias、单头、LayerNorm、Xavier；见 §4.2 |
| 学习率倍率形式、KVB reduction 和 mini-batch 细节 | 使用本文实现方案，并在配置中明确默认值 |
| TBPTT 长度、outer step 频率、mask 后 loss reduction | 使用本文策略，显式记录分段与归一化规则 |
| H、stride、去噪步数和状态提交方式 | 从固定基座和任务定义读取；论文未明部分显式配置 |
| curriculum、WSD 日程、精度和裁剪 | 完整写入配置，不将开发默认值冒充论文配方 |
| 纯视频缺失动作/状态的表示 | 使用显式 modality mask；核对论文/官方实现后更新 |

已完成语法、导入、配置、安装脚本语法与临时 HDF5 IO/掩码/统计检查。开发主机现有 Python/PyTorch 版本与目标版本不同，且没有 CUDA；未下载完整权重或真实数据集，未做 GPU 前向/反向、训练、checkpoint 实际恢复或模拟器运行。交付代码与入口不等于这些端到端过程已通过。按用户要求不包含实验结果报告。

## 11. 参考来源

- **[R1]** [RoboTTT 官方项目主页](https://research.nvidia.com/labs/gear/robottt/)。
- **[R2]** [RoboTTT 论文全文](https://research.nvidia.com/labs/gear/robottt/robottt.pdf)，重点为 §2–4、图 2/4/8、附录 A–B。
- **[R3]** [NVIDIA Isaac-GR00T 官方 README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/README.md)，用于公开版本及接口核对。
- **[R4]** [官方 GR00T-N1.7-3B 模型卡](https://huggingface.co/nvidia/GR00T-N1.7-3B)。
- **[R5]** [官方 checkpoint config](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/main/config.json)。
- **[R6]** [官方 DiT 实现](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/modules/dit.py)。
- **[R7]** [官方 N1.7 模型与动作头](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d7/gr00t_n1d7.py)。
- **[R8]** [官方 N1.7 processor](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py)。
- **[R9]** [PyTorch autograd.grad](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.grad.html)，用于可微内循环 API 依据。
- **[R10]** [PyTorch functional_call](https://docs.pytorch.org/docs/2.9/generated/torch.func.functional_call.html)，函数式调用参考；当前实现直接使用显式张量和 batched matmul。
- **[R11]** [RoboMME-Plus 数据格式](https://github.com/RoboMME/robomme_benchmark/blob/1f08ca5a4573eda164906dbfcf2f3511a7b62e90/doc/plus_training_data_format.md)。

以上动态链接记录核查入口；实际实现保存不可变 commit/revision，避免后续上游更新改变依赖条件。
