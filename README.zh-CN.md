# PoseCascade

> **语言**：[English](README.md) · [繁體中文](README.zh-TW.md) · **简体中文**
> **文档**：[Read the Docs 源代码](docs/)（Sphinx）

一个基于 PySide6 + OpenGL 的桌面引擎，用于导入 3D 模型并以沙箱化脚本驱动它们。
视觉目标是 MMD 风格：卡通着色 + 锐利的 inverted-hull 描边、带 sphere map 的 PMX 材质、
VMD 风格动画曲线、IK + 脚掌贴地、morph target，以及可实时摆动数件布料的 PBD 布料求解器。

## 功能特色

- **多格式导入**：glTF / GLB、OBJ + MTL、FBX、STL、PLY、USD / USDZ、COLLADA，
  以及 MMD 三件套（PMX 模型 + VMD 动作 + VPD 姿势）。每个格式都挂在自己的插件 adapter
  后面，新增格式不会动到渲染器。
- **MMD 味前向渲染器**：toon ramp（NEAREST + clamp 取得干净色块边界）、sphere-map
  合成、inverted-hull 描边、程序化棋盘地面、投影地面阴影、深度贴图 PCF-softened
  自我阴影、sRGB-aware 输出、渐变天空 pass、多光源 HighDef 配置（1 主光 + 最多 3 副光）、
  可选的对偶四元数蒙皮（保持关节体积）、默认 AutoLuminous bloom、MMD tone-curve 后处理、
  程序化舞台抽象（地板 + 背墙 + 侧墙），以及一个 selection-overlay pass，会用对比色把
  选中的 top-level holder 再描一次边。完整 pass 顺序与每个 pass 的开关见
  [`docs/rendering_pipeline.md`](docs/rendering_pipeline.md)。
- **VMD 驱动动画**：per-bone / per-morph / per-camera 轨道，使用 MMD 惯用的四控制点
  贝塞尔插值、IK 求解器、脚掌贴地器、slot 之间的 external-parent 绑定、display-frame
  分组以及 physics 链。
- **布料求解器**：position-based dynamics 含结构约束 + bend 约束、球体 / 胶囊碰撞体
  （含连续碰撞 sweep），以及一个 Cython kernel,在 480-vert 裙子上把每步成本压到
  **0.35 ms** 左右——比纯 NumPy fallback 快约 9×。若编译扩展未构建,kernel 会透明
  地退回 NumPy 路径。
- **声明式动画 runtime**：用 JSON 文档而不是 Python 代码来驱动角色——phases、
  gaits、body trajectories、morph 时间线,以及行内 expression DSL。
  详见 [`docs/declarative_animation.md`](docs/declarative_animation.md)。
- **沙箱化 Python 脚本**：在受限的命名空间里执行（拿不到 `open`、`os`、`subprocess`、
  `__import__` …）,对外只提供 `scene`、`nodes`、`time`、`input`,以及一层整理过的
  数学 API,让用户摆姿势、打 keyframe、做动画,不必碰引擎内部。
- **MCP 服务器**：一个 Model Context Protocol 服务器,让任何支持 MCP 的 LLM agent
  都能驱动引擎——列出并验证声明式动画脚本、检查模型、跑布料 benchmark。详见
  [`docs/mcp.md`](docs/mcp.md)。

## 快速开始

```bash
# Clone + 建立虚拟环境
git clone https://github.com/JeffreyChen-s-Utils/PoseCascade.git
cd PoseCascade
python -m venv .venv
.venv\Scripts\activate.ps1            # Windows PowerShell
# source .venv/bin/activate           # Linux / macOS

# 带 AI extras 安装,这样连 MCP 服务器一起装起来
pip install -e .[dev,ai]
```

editable install 会就地编译 Cython 布料 kernel。没有 C 编译器
（Windows 上的 Microsoft Build Tools、Linux / macOS 上的 gcc / clang）时,
安装会打印警告,引擎透明退回 NumPy 路径。

针对随附示例启动编辑器：

```bash
# 经典 3D 模型展示轮播（30 秒）：intro idle → 360° turntable →
# 原地走路 → 挥手 → V-pose → hip-pop → 鞠躬 → 回到 neutral。
python -m posecascade --scene examples/assets/herta/herta.glb \
                       --script examples/scripts/showcase.json

# 原地行走 + 摆手 4 秒循环。
python -m posecascade --scene examples/assets/herta/herta.glb \
                       --script examples/scripts/walk.json

# 最小呼吸 idle 4 秒循环。
python -m posecascade --scene examples/assets/herta/herta.glb \
                       --script examples/scripts/idle.json
```

或者 headless 直接渲染 MMD 风格 hero frame——不开编辑器窗口,
适合当整套视觉管线的 smoke check：

```bash
# 加载 herta.glb（glTF）,在 force_toon 路径上开所有 MMD-fluence
# 开关,输出 mmd_demo.png 到脚本旁边。
python examples/mmd_demo.py

# 加载随附的 March 7th PMX,走渲染器的 PMX-native 路径
# （per-mesh MMDMaterial + sphere texture + 直接从文件读的 edge flag）。
# 这个素材是第三方资源,授权说明见 examples/assets/march7th/NOTICE.md。
# 输出 march7th_pmx_demo.png 到脚本旁边。
python examples/march7th_pmx_demo.py

# 也可以通过交互式编辑器执行：
python -m posecascade --scene examples/assets/march7th/march7th.pmx
```

一组并排对比脚本展示每个 MMD-fluence 功能 vs baseline——
每个都会输出标好标签的 PNG,让视觉差异可重现：

```bash
python examples/compare_bloom.py    # bloom 关闭 vs 应用 AutoLuminous
python examples/compare_tone.py     # 纯 sRGB vs + mmd_tone
python examples/compare_dqs.py      # LBS 糖果包装 vs DQS 极限扭转
python examples/compare_lights.py   # 只开主光 vs + HighDef rim + fill
```

## 项目结构

```
PoseCascade/
├── posecascade/                  # 主包
│   ├── animation/                # 布料、蒙皮、morph、IK、VMD 轨道
│   │   ├── cloth.py              # PBD 求解器（Python 编排层）
│   │   └── _cloth_kernels.pyx    # Cython 内循环（由 setup.py 构建）
│   ├── app/                      # QApplication bootstrap、主窗口
│   ├── assets/                   # 缓存、路径安全、importer 管理器
│   ├── gl/                       # GL context、shader、framebuffer
│   ├── mcp/                      # Model Context Protocol 服务器
│   ├── render/                   # 渲染图、材质、灯光
│   ├── scene/                    # scene graph、transform、component
│   ├── scripting/                # 沙箱脚本主机 + 声明式 runtime
│   └── ui/                       # viewport、outliner、inspector、timeline
├── importers/<format>/           # 每个格式各自的 importer 插件
├── shaders/                      # 按 render pass 分类的 GLSL
├── examples/                     # 随附模型 + 动画脚本
├── tests/                        # pytest 测试套件,镜像包结构
├── docs/                         # 设计 + 集成文档
├── schemas/                      # JSON schema（声明式动画）
├── setup.py                      # cythonize build hook
└── pyproject.toml                # 项目 metadata + ruff / bandit 配置
```

## 开发

[`CLAUDE.md`](CLAUDE.md) 里的 Definition of Done 要求每个变更在 commit 之前
都要通过三道闸门。本地重现：

```bash
.venv/Scripts/python.exe -m pytest tests/             # unit + offscreen-GL 测试
.venv/Scripts/python.exe -m ruff check .              # lint + style
.venv/Scripts/python.exe -m bandit -c pyproject.toml -r posecascade/
```

布料 Cython kernel 每次改 `.pyx` 源码后都要重新就地构建：

```bash
.venv/Scripts/python.exe setup.py build_ext --inplace
```

要发布的时候,`pyproject.toml` 里的 `[tool.cibuildwheel]` 段会在
Win / macOS / Linux × 各支持 Python 版本上产出预构建 wheel;
`.github/workflows/wheels.yml` 在 tag push 时触发那个流程。

## 视觉管线

前向渲染器每帧按固定顺序跑六个 pass —— depth-map 阴影 pass、scene、地面、
projected shadow、selection overlay、后处理 effect chain。每个 pass 都有开关
（`set_ground_enabled`、`set_self_shadow_enabled`、`set_projected_shadow_enabled`、
`set_selected_holder`）,这样 smoke test 与 headless render 可以选择关掉某些 pass
而不影响其他 pass 的像素保真度。完整拆解——pass 顺序、shader 文件、light-space
数学、texture unit、MMD-fluence gap——都在
[`docs/rendering_pipeline.md`](docs/rendering_pipeline.md)。

## 性能备注

布料求解器是近期主要的优化重点。`posecascade.mcp.server.cloth_benchmark`
里 480-vert 裙子的 benchmark：

| 阶段                                   | ms/step (best) |  vs baseline |
|----------------------------------------|---------------:|-------------:|
| Baseline（优化前）                     |          3.225 |            — |
| NumPy：einsum + 合并 bincount          |          2.085 |         −35% |
| Cython kernel                          |          0.356 |     **−89%** |
| Cython + broad-phase + bin culling     |  0.36–0.38（单 bin collider 多省 30%） | — |

渲染器热路径的每帧计时都用 `posecascade.utils.profiling.frame_section` 包起来,
UI overlay（或自定义测试）能从 `current_stats().sections` 拉出每帧分解。

## 许可

项目的 MIT-style 条款见 [`LICENSE`](LICENSE)。随附素材各有自己的许可——
`examples/assets/herta/herta.glb` 以 CC-BY 4.0 分发
（上传者为 Sketchfab 上的 X9_YT；角色 “The Herta” © HoYoverse,
依其 Fan Content Guidelines 使用——完整声明见
`examples/assets/herta/NOTICE.md`）。MMD 演示
`examples/assets/march7th/march7th.pmx` 单独以 CC-BY 4.0 分发
（上传者 Gregman；角色 “March 7th” © HoYoverse）——见
`examples/assets/march7th/NOTICE.md`。
