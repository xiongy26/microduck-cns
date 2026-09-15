# MALE CNS / MICRODUCK — connectome-driven robot dashboard

复现截图效果的 Web demo：用「果蝇雄性中枢神经系统（Male CNS）connectome 子集」的
**仿真神经动力学** 探索性地驱动 MicroDuck 双足机器人的行走、转向与头部扫描运动，
左侧是机器人仿真、右侧是神经解剖映射，底部是 14 关节的控制器输出时间轴。

机器人低层运动由**真实训练 RL 策略 + MuJoCo 物理仿真**执行
（MuJoCo WASM + onnxruntime-web，策略为 `BEST_alpha_stand/walking.onnx`），
**大脑（connectome CTRNN）只出高级
指令**——前进速度、转向速率和头部关节目标偏移，类似真实苍蝇"脑→VNC 下行命令"
的分层控制。

![screenshot](docs/screenshot.png)

## 运行

```bash
cd microduck-cns
python3 server.py        # 带 no-cache 头的静态服务器, 端口 8123
# 打开 http://127.0.0.1:8123
```

无任何外部依赖（three.js 已本地化在 `assets/vendor/`），离线可用。

## 数据来源与致谢

| 部分 | 来源 |
|------|------|
| 机器人模型 | [Pollen Robotics **MicroDuck**](https://github.com/pollen-robotics/microduck)（开源双足鸭子机器人）— 关节定义取自 [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl) 的 MJCF（`robot_walk.xml`，14 关节），网格为官方 STL |
| 神经系统原型 | Janelia FlyEM 的 **MANC**（雄性成虫神经索 connectome，~25,000 神经元 / [neuprint-cns.janelia.org](https://neuprint-cns.janelia.org)）与 FAFB 雄性大脑（FlyWire） |
| 神经可视化数据 | **程序化生成**（见下），但数量契约与真实一致：192 神经元 / 2,456 连接，按真实解剖分区（脑 56、下行 16、VNC 运动池 96、中间 8、上行 16），运动池→关节为拓扑映射（前腿关节→ pro/meso/metathoracic 神经节，头颈关节→ 食道下神经节 SOG） |
| 控制器 | **方案 B 分层控制**：192 神经元 CTRNN 微回路（模式/转向/扫描 + 感觉反馈）输出高级指令；低层平衡与步态由训练好的 RL 策略执行。连接为结构化功能微回路 + 弥散弱耦合（参数经 tools 外的原型扫描调定），非训练 MANC 权重 |

## 神经活动 → 机器人运动的数据流（闭环）

```
        ┌──────────────────────── 192 神经元 CTRNN（connectome.js）──────────────────────┐
        │  感觉上行(16) ← 身体状态: 速度/偏航率/直立度/摔倒/头颈本体感觉                  │
        │  脑(56): 模式回路 W/S 翻转触发器 + 稳态压 → 走/停节律                          │
        │          转向回路 L/R + 稳态压 → 探索性转向；扫描振荡器 SY/SP → 头部扫描        │
        │  下行(16): DW/DS/DL/DR 指令群 ──解码──► vx / wz / 头部关节偏移                 │
        └──────────────────────────────────┬────────────────────────────────────────────┘
                                           ▼
  RL 策略(ONNX, 50Hz) ◄─ 指令(vx,wz)   头部偏移加在策略动作上
        │ 观测: 陀螺仪/重力投影/关节角/关节速/上一动作/指令(61维)
        ▼
  MuJoCo 物理(200Hz) ──► qpos ──► 机器人渲染 + 关节速度/动作 ──► VNC 运动池活性(着色)
        │
        └──► 身体状态回灌大脑感觉通道（闭环）
```

- 大脑是真正的**在环控制器**：每个渲染帧积分全部 192 个神经元的 CTRNN 动力学
  （2456 条加权连接 + OU 噪声 + 4 个慢稳态变量），指令从下行神经元群解码——
  `vx = 0.25·(DW−DS)`、`wz = 0.35·(DL−DR)`、头部偏移来自扫描振荡器群
- 走/停节律由 W/S 互相抑制（赢者通吃）+ 慢稳态压（行走时积累"疲劳"，站立时
  积累相反压力）产生：行走 bout ≈ 6 s、站立 bout ≈ 3–5 s，类似果蝇的探索-停顿节律
- 摔倒时感觉通道"fallen"强激活 S 群、抑制 W 群 → 大脑立即停止行走指令，
  配合物理端的摔倒恢复状态机（stand 策略自我扶正）
- 视觉着色与原截图语义一致：运动池活性 = 实际肌张力读出（青=正/橙=负），
  但现在它是**脑动力学的一部分**（运动池接受下行指令与本体感觉输入）
- CONNECTOME DYNAMICS 开关：2,456 条连接 + 沿真实活动突触行进的脉冲动画
- HUD 的 *RL policy / mode* 显示策略与大脑当前行为模式；关掉
  `TRAINED CONTROLLER` 芯片 = 消融实验（未训练噪声控制器，物理暂停）

## 交互

- **拖拽/滚轮**：机器人视口自由 orbit/zoom（相机跟随机器人）
- **底部芯片开关**：
  - `WIREFRAME` 机器人 X 光线框
  - `CONNECTOME DYNAMICS` 显示 2,456 条连接 + 行进中的突触脉冲
  - `TRAINED CONTROLLER` ON = RL 物理行走/站立；OFF = 未训练噪声控制器
  - `PHYSICS` / `OCEAN FLOOR` 视觉效果开关（RL 模式下物理恒开）
- **时间轴**：20 秒环形缓冲；拖动进度条可在暂停时回放关节角度
- ▶/❚❚ 播放暂停，⛶ 全屏
- 注意：onnxruntime-web 从 CDN 加载（约几 MB），首次打开需联网

## 文件结构

```
microduck-cns/
├── index.html / css/style.css      # 仪表盘布局
├── js/
│   ├── main.js                     # 三视口渲染 / HUD / 开关 / 状态同步
│   ├── physics.js                  # MuJoCo WASM + RL 策略推理（执行大脑指令）
│   ├── robot.js                    # MJCF 运动学树重建 + STL 材质分配
│   ├── gait.js                     # (备用) 过程式步态 + 未训练消融模式
│   ├── connectome.js               # 192 神经元 CTRNN 大脑：微回路动力学 + 指令解码 + 解剖渲染
│   └── timeline.js                 # 14 关节 20s 环形缓冲时间轴
├── assets/
│   ├── robot.json                  # 由 tools/mjcf_to_json.py 转换的运动学树
│   ├── meshes/*.stl                # 官方机器人视觉网格 (来自 microduck_rl)
│   ├── mjcf/                       # 物理模型: robot_allcollisions.xml + 碰撞网格
│   ├── policies/*.onnx             # 训练好的 RL 策略 (stand / walking)
│   └── vendor/                     # three.js r180 + MuJoCo WASM + loaders
└── tools/
    ├── robot_walk.xml              # 原始 MJCF (存档)
    └── mjcf_to_json.py             # MJCF → robot.json 转换器
```

## RL 物理管线（来源与致谢）

站立/行走的策略推理与物理仿真管线移植自本地项目
`~/humanoid-robot/microduck-ar`（MuJoCo WASM + onnxruntime-web 的浏览器端
sim2sim 部署），策略文件为该项目 `public/policies/` 下的
`BEST_alpha_stand.onnx` 与 `BEST_alpha_walking.onnx`（"alpha" 即 MicroDuck
在 mjlab 训练管线中的代号，模型定义同源于 `pollen-robotics/microduck_rl`）。
控制器数值（指令范围、PD、STAND 关键帧、摔倒恢复状态机）与该部署 1:1。

## 换入真实 MANC 数据（可选）

`connectome.js` 中预留了替换点。用 [neuprint-python](https://github.com/connectome-neuprint/neuprint-python)
（需在 neuprint-cns.janelia.org 免费注册 token）：

```python
from neuprint import Client, fetch_skeleton
c = Client('https://neuprint-cns.janelia.org', dataset='manc', token='YOUR_TOKEN')
skel = fetch_skeleton(bodyId= bodies[0])   # SWC 样式骨架
```

把 192 个骨架（SWC 折线）与突触连接矩阵写入 `assets/connectome.json`，
在 `Connectome.growAll()/wire()` 里优先加载该文件即可 — 其余渲染、着色、
池映射逻辑无需改动。
