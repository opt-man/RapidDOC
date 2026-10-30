# RapidDOC

按最终论文的三个实验组织。每个实验的实现、输入数据、结果和图放在同一目录；共用求解器源码放在 `rapidoc/`。

```text
RapidDOC_artifact_20260920/
├── README.md
├── requirements.txt                 Python 依赖清单
├── rapidoc/                         共用求解器
│   ├── factor_reuse_solver.cpp
│   ├── factor_reuse_solver.h
│   ├── solver.py                    Python 接口
│   ├── CMakeLists.txt
│   └── patches/                     FATROP 缓存导出改动
├── experiments/
│   ├── exp1_factor_reuse_ablation/  实验一：因子复用消融
│   │   ├── experiment.py           三组消融的实验实现
│   │   ├── frozen_cases.py         实例读取
│   │   ├── linear_system.py        线性系统与求解接口
│   │   ├── plot.py                 Fig. 2 绘图
│   │   ├── data/                   三个任务，各 32 个固定实例
│   │   ├── results/                原始计时、统计与配置
│   │   └── figures/                Fig. 2：PDF / PNG / SVG
│   ├── exp2_ocp_imitation/          实验二：开环、闭环模仿学习
│   │   ├── imitation_learning.py   训练与绘图实现
│   │   ├── imitation_models.py     CartPole / Quadrotor / Robot Arm
│   │   ├── baselines/              CasADi、acados 适配及计时接口
│   │   ├── native/                 C++ 接口及生成的模型源码
│   │   ├── patches/                baseline 必需的定制改动
│   │   ├── data/                   冻结的示范数据
│   │   ├── results/
│   │   │   ├── open_loop/          开环学习记录与最终参数
│   │   │   └── closed_loop/        闭环学习记录与最终策略
│   │   └── figures/                Fig. 3 / Fig. 4
│   └── exp3_vehicle/               实验三：小车学习与实物验证
│       ├── training/               原生闭环训练代码与模型
│       ├── ros/                    实车 ROS 控制代码
│       ├── plot_learning.py        Fig. 5 绘图
│       ├── video_composite/        Fig. 6 真实视频帧合成代码
│       ├── data/                   训练日志和实测轨迹
│       ├── results/                轨迹选择及合成记录
│       └── figures/                Fig. 5 / Fig. 6
└── paper/
    ├── RapidDOC.pdf                最终论文
    ├── RapidDOC.tex                论文源码
    ├── references.bib
    ├── ieeeconf.cls
    ├── figures/                    论文引用的六张 PDF 图
    └── ICRA2027_RapidDOC.mp4        投稿视频
```

## 论文结果对应关系

| 内容 | 位置 |
|---|---|
| Fig. 2，消融实验 | `exp1_factor_reuse_ablation/results/` |
| Fig. 3，开环学习 | `exp2_ocp_imitation/results/open_loop/` |
| Fig. 4 与 Table I，闭环学习 | `exp2_ocp_imitation/results/closed_loop/` |
| Fig. 5 与 Table II，小车学习 | `exp3_vehicle/data/` |
| Fig. 6，实物残影 | `exp3_vehicle/figures/` 与 `results/` |

表中位置均相对于 `experiments/`。结果文件中的历史标识 `Ours` 对应论文中的 RapidDOC，原始数据保持原样。

开环为 seed 2、800 次迭代、学习率 `2e-3`；闭环为 seed 2、500 次迭代、50 个 MPC 步、学习率 `5e-3`。两者均为 `N=40`、`dt=0.05`。消融为每个任务 32 个实例，具体计时边界见结果目录内的 JSON 配置。

## 代码与依赖

保留的是实际实验实现和绘图代码，没有额外的 PowerShell / Shell 启动包装脚本。Python 依赖见 `requirements.txt`；FATROP、acados、CasADi/IPOPT 原生运行库和 ROS 单独安装。`patches/` 中是实验必需的源码改动，不是安装包。

`rapidoc/patches/fatrop_cache_export.patch` 对应 FATROP 基准提交 `5951e14989175108e0a64c0f726af8fbe19563f3`。共用求解器由 `rapidoc/CMakeLists.txt` 构建；模仿学习原生接口由该实验的 `native/CMakeLists.txt` 构建，需指定已修改的 FATROP 源码和安装路径。图表重绘与数据查看不需要加载这些原生库。

原始大视频保留在原工程；此目录收录最终投稿视频、合成代码、选帧记录和成图。各图的最终导出保存在 `figures/`，新绘图输出单独保存。
