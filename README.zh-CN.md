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
  gaits、body trajectories、morph 时间线、行内 expression DSL，加上 `extends`
  继承机制与数组简写（`[from, to]` 曲线、`[x, y, z]` 平移、bones 的 `x`/`y`/`z`
  轴别名），把典型动画文件长度压到原本长写法的 1/3。
  详见 [`docs/declarative_animation.md`](docs/declarative_animation.md)。
- **编辑器内动画编辑器**（新）：两个共用同一份 in-memory document 的右侧 dock——
  JSON 编辑器具备语法着色、行号 gutter、parse error 行内标记、Format 按钮、
  dirty indicator；Phase 方块 dock 则有横向时间轴（拖拽重排 + 拖边缘调 duration）、
  纵向卡片列表，加上 inline 表单覆盖所有常用字段（name / duration / blends /
  pose / gait / body translation / bones / morphs）。两边都接 Ctrl+Z / Ctrl+Y
  undo/redo。详见 [`docs/animation_editor.md`](docs/animation_editor.md)。
- **GPU compute 蒙皮**（新）：用 OpenGL 4.3 compute shader 把 `passive_skin_deform`
  布料的 LBS + 碰撞推开 + world-to-local 都搬到 GPU，直接写进 mesh 的 position
  与 normal VBO。在 30k-vert 身体 mesh 上把每帧 cloth + apply_cloth 从 ~9 ms 降到
  0.05 ms 以下；GL 版本低于 4.3 或 compute 编译失败时透明退回 CPU LBS 路径。
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

## 持续集成 + 自动发布

三个 GitHub Actions workflow 串起整条 pipeline：

- **`tests.yml`** 在每个 PR 跟每次 push 到 `main` 跑 `ruff` + `bandit` +
  完整 `pytest`（透过 `xvfb-run` 让 Qt 跑得起来）。三个 Python 版本
  （3.12 / 3.13 / 3.14）在 Ubuntu 上。
- **`release.yml`** 在每次 push 到 `main` 跑—— 读最新的 `v*` git tag,
  bump patch 版本（commit subject 含 `[release major]` / `[release minor]`
  则改成对应的 bump）,然后 push 新 tag。docs-only PR 想跳过时在 merge
  commit message 里写 `[skip release]`。
- **`wheels.yml`** 在新 tag push 时触发,透过 `cibuildwheel` 跑全 OS ×
  Python 矩阵、跑 in-wheel smoke test,然后透过 Trusted Publishing
  **自动发布到 PyPI**—— repo 不需要存任何 API token。

整体效果：merge 一个 PR 就会自动发成一版新的 PyPI release。版号透过
setuptools-scm 读刚 push 进去的 tag 取得,`pyproject.toml` 的
`dynamic = ["version"]` 开启这个机制。一次性的 PyPI Trusted Publishing
配置、bump 消息规则、yank / revert 指引、本地 dry-run 食谱,都在
[`docs/release_pipeline.md`](docs/release_pipeline.md)。

## 视觉管线

前向渲染器每帧按固定顺序跑六个 pass —— depth-map 阴影 pass、scene、地面、
projected shadow、selection overlay、后处理 effect chain。每个 pass 都有开关
（`set_ground_enabled`、`set_self_shadow_enabled`、`set_projected_shadow_enabled`、
`set_selected_holder`）,这样 smoke test 与 headless render 可以选择关掉某些 pass
而不影响其他 pass 的像素保真度。完整拆解——pass 顺序、shader 文件、light-space
数学、texture unit、MMD-fluence gap——都在
[`docs/rendering_pipeline.md`](docs/rendering_pipeline.md)。

## 性能备注

最近两条主要的优化路线：布料求解器与渲染器热路径。

**布料** —— 480-vert 裙子 benchmark
（`posecascade.mcp.server.cloth_benchmark`）：

| 阶段                                   | ms/step (best) |  vs baseline |
|----------------------------------------|---------------:|-------------:|
| Baseline（优化前）                     |          3.225 |            — |
| NumPy：einsum + 合并 bincount          |          2.085 |         −35% |
| Cython kernel                          |          0.356 |     **−89%** |
| Cython + broad-phase + bin culling     |  0.36–0.38（单 bin collider 多省 30%） | — |

showcase 场景中 30k-vert 身体 mesh 那条 passive_skin LBS + 碰撞推开,
GPU compute 路径把原本 ~9 ms 的 CPU 成本压到 0.05 ms 以下,直接写进 mesh
既有的 position / normal VBO,没有 CPU readback。

**渲染器** —— 768×768 showcase 场景开全部 pass（shadow + projected shadow
+ ground + outline + toon）,用 `tools/bench_renderer.py` 跑：

| 阶段                                   | ms/frame |    FPS |
|----------------------------------------|---------:|-------:|
| Baseline                               |     7.88 |    127 |
| 每帧 uniform-state 缓存                |     6.09 |    164 |
| + shadow pass 拉出 glUseProgram        |     5.04 |    198 |
| + 批次 bone-matrix matmul              | **~4.50**| **~220**|

收益来源：per-program「这帧已上传过」的 uniform 缓存省下大约 85% 的
`glUniformMatrix4fv` / `ascontiguousarray` 调用；把 `glUseProgram` 拉出
depth 与 projected-shadow pass 的 per-mesh 内循环；以及把
`_compute_bone_matrices` 里每个关节的 Python matmul 换成单一批次 `np.matmul`。

**声明式 runtime** —— 每帧 parent-world rotation 缓存、`lru_cache` 化的
expression AST 解析、以及只在加载时跑的 `extends` 解析（没有 per-frame 成本）。
在 showcase 上 JSON 驱动的 update 稳定在 0.6 ms 左右。

渲染器热路径的每帧计时都用 `posecascade.utils.profiling.frame_section` 包起来,
UI overlay（或自定义测试）能从 `current_stats().sections` 拉出每帧分解。
`tools/bench_renderer.py` 把 showcase 场景跑指定帧数并打印细目—— renderer 改动
push 之前可以先用它抓 regression。

## 许可

项目的 MIT-style 条款见 [`LICENSE`](LICENSE)。随附素材各有自己的许可——
`examples/assets/herta/herta.glb` 以 CC-BY 4.0 分发
（上传者为 Sketchfab 上的 X9_YT；角色 “The Herta” © HoYoverse,
依其 Fan Content Guidelines 使用——完整声明见
`examples/assets/herta/NOTICE.md`）。MMD 演示
`examples/assets/march7th/march7th.pmx` 单独以 CC-BY 4.0 分发
（上传者 Gregman；角色 “March 7th” © HoYoverse）——见
`examples/assets/march7th/NOTICE.md`。
