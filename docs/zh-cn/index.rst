PoseCascade 使用手册
====================

一个基于 PySide6 + OpenGL 的桌面引擎，用于导入 3D 模型并以沙箱化脚本驱动它们。
视觉目标是 **MMD**：卡通着色 + 锐利的 inverted-hull 描边、带 sphere map 的
PMX 材质、VMD 风格动画曲线、IK + 脚掌贴地、morph target，以及可实时摆动数件
布料的 PBD 布料求解器。

.. contents:: 目录
   :depth: 2
   :local:

----

安装
----

PoseCascade 针对 Python **3.14**，支持 Windows、macOS、Linux。建议使用项目内的
virtualenv，这样 editable install 编译 Cython 布料 kernel 时能对到同一个 interpreter：

.. code-block:: bash

   git clone https://github.com/JeffreyChen-s-Utils/PoseCascade.git
   cd PoseCascade
   python -m venv .venv
   .venv\Scripts\Activate.ps1            # Windows PowerShell
   # source .venv/bin/activate           # Linux / macOS

   pip install -e .[dev,ai]

``dev`` extra 会拉进 ``pytest``、``pytest-qt``、``ruff``、``bandit`` 与
``scikit-image``（golden-image SSIM 容差用）。``ai`` extra 会安装可选的 ``mcp``
服务器入口。

.. note::
   editable install 会就地编译 ``posecascade/animation/_cloth_kernels.pyx``。
   Windows 需要 **Microsoft Build Tools**；Linux / macOS 需要 gcc 或 clang。
   没有 C 编译器的话，``pip`` 会打印警告，引擎会透明退回 NumPy fallback 路径——
   速度比较慢，但功能上完全一致。

可选依赖
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Extra
     - 解锁什么
   * - ``[ai]``
     - ``mcp`` + ``jsonschema``——安装 ``posecascade-mcp`` CLI，
       供 Claude Code、Claude Desktop、Cursor … 使用。
   * - ``[fbx]``
     - FBX importer 插件（独立 runtime 因为 FBX SDK 是 Autodesk 授权且体积大）。
   * - ``[usd]``
     - Universal Scene Description 插件（Pixar 的 USD bindings）。
   * - ``[collada]``
     - ``.dae`` importer（XML-based；用 ``defusedxml`` 防 XXE 攻击）。

----

运行示例
--------

随附示例放在 ``examples/``。分三类：

交互式 viewport（声明式 JSON）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最简单看到角色动起来的方式。``--scene`` 传模型，``--script`` 传 ``.json``
动画文档：

.. code-block:: bash

   # 原地走路 + 摆手 —— 4 秒循环
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/walk.json

   # 30 秒展示轮播 —— idle → turntable → wave → V-pose → bow
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/showcase.json

   # 上下楼梯（声明式 phases）
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/climb_stairs.json

   # 最小呼吸 idle 循环
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/idle.json

   # 四足"狗爬式"——先摆到双手双膝着地，再向前爬行
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/dog_crawl.json

交互式 viewport（沙箱 Python）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

把扩展名换成 ``.py`` 就会加载沙箱 Python 脚本而不是 JSON 文档。
适合 per-frame 逻辑不适合塞进 phases 的情景（例如纯布料演示、头发物理示范）：

.. code-block:: bash

   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/walk.py
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/hair_sway.py
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/cape_cloth.py
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/full_demo.py
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/idle_orbit.py

Headless / smoke render
^^^^^^^^^^^^^^^^^^^^^^^

独立脚本，不开 Qt 窗口——它们开一个 offscreen GL context、渲染单张 frame
（或 N 张的 strip），然后把结果写到磁盘。适合当整个视觉管线的快速 smoke check：

.. code-block:: bash

   # MMD 风格 hero frame，每个 fluence toggle 都开
   python examples/mmd_demo.py

   # PMX-native 路径（per-mesh MMDMaterial + sphere texture）
   # --frames N 会输出 N 张的动画条
   python examples/march7th_pmx_demo.py --frames 8

   # 并排比较渲染
   python examples/compare_bloom.py     # AutoLuminous bloom 关 vs 开
   python examples/compare_dqs.py       # LBS 糖果包装 vs DQS
   python examples/compare_lights.py    # 只开主光 vs HighDef 多光
   python examples/compare_tone.py      # 纯 sRGB vs + mmd_tone

   # 360° 转场（输出 spin.mp4）
   python examples/spin.py

随附角色
^^^^^^^^

所有随附示例都使用 ``examples/assets/herta/herta.glb``
（《崩坏：星穹铁道》的「The Herta」——354 个 joint，连衣裙绑入身体 mesh）。
MMD 演示路径用另一个 PMX 资产：

.. code-block:: bash

   python -m posecascade --scene examples/assets/march7th/march7th.pmx

两个素材都是第三方 CC-BY 4.0。glTF 上传者为 Sketchfab 上的 X9_YT
（角色「The Herta」© HoYoverse，依其 Fan Content Guidelines 使用）；
PMX 上传者为 Sketchfab 上的 Gregman（角色「March 7th」© HoYoverse）。
完整声明分别见 ``examples/assets/herta/NOTICE.md`` 与
``examples/assets/march7th/NOTICE.md``。

----

编写声明式动画
--------------

声明式动画是放在 ``examples/scripts/`` 下的 JSON 文档，把角色绑进一连串的 **phases**。
每个 phase 指定持续时间、身体做什么（translation、yaw、lean）、肢体做什么
（gait、IK target、pose preset），以及 morph 做什么（smile、blink、mouth-A）。
Runtime 会在 phases 之间 cross-fade，跑到结尾自动 loop，把每帧的 pose 喂给渲染器。

文档结构
^^^^^^^^

.. code-block:: json

   {
     "schema_version": 1,
     "name": "walk_in_place_demo",
     "loop_sec": 4.0,
     "rig": {
       "character_root": "Sketchfab_model",
       "body_bones": {
         "head":        "Head_M_055",
         "upper_arm_L": "Shoulder_L_0183",
         "upper_arm_R": "Shoulder_R_0233"
       }
     },
     "phases": [
       {
         "name": "walk",
         "duration_sec": 4.0,
         "body":  { "yaw_rad": 0.0 },
         "gait":  { "kind": "walking", "step_cycle_sec": 1.0,
                    "leg_swing_amplitude": 0.50,
                    "arm_swing_amplitude": 0.55 }
       }
     ]
   }

顶层 key：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - 用途
   * - ``schema_version``
     - 永远是 ``1``。未来不向下兼容的变更会 bump。
   * - ``name``
     - log 与 MCP ``list_animations`` 工具用的稳定识别码。
   * - ``loop_sec``
     - 一个 loop 的总长度。必须等于所有 phase 的 ``duration_sec`` 总和
       （parser 会检查）。
   * - ``rig``
     - 骨头命名别名——见下面的 *骨头别名*。
   * - ``ground``
     - 可选的平地面（Y 高度），给脚掌 IK 夹钳用。
   * - ``phases``
     - 有顺序的列表。每个项目跑 ``duration_sec`` 秒。
   * - ``hide``
     - 可选列表，列出开始时要从 scene 取下的 node 名
       （动画不想要的 prop、灯光、楼梯）。
   * - ``cloth`` / ``colliders``
     - 可选的声明式布料设置——见下面的 *布料*。
   * - ``pose_library``
     - 可选的文档内 pose preset，会覆盖同名 built-in。

Phase 内容
^^^^^^^^^^

每个 phase 可以声明下列一个或多个字段：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 驱动什么
   * - ``body``
     - 整个角色的 yaw、translation、lean。值是 expressions，每帧求值
       （``"phase_t"`` 是 phase 内已过的秒数）。
   * - ``gait``
     - 程序化的走 / 跑 cycle。用解析公式设定腿 + 手臂角度。
       ``kind`` 可以是 ``"walking"``、``"running"``、``"idle"``。
   * - ``ik``
     - 脚掌 IK 目标。配合 ``ground`` 夹钳能让脚掌一直在地板上。
   * - ``poses``
     - 命名 preset 混合（``reach_R_soft``、``wave_L``、``T_pose`` …）。
       Built-in preset 在 ``posecascade/scripting/pose_library.py``。
   * - ``morphs``
     - per-morph 权重，支持 expression
       （``"smile": "0.5 + 0.5*sin(phase_t * pi)"``）。
   * - ``cross_fade_sec``
     - 从前一个 phase 缓进到这个 phase 的秒数。Runtime 会插值 pose + morph 权重。

Expression DSL
^^^^^^^^^^^^^^

数值字段接受 literal 数值或 string expression。DSL 是 Python 的安全子集——
没有 ``import``、没有 attribute access、没有 function definition。可用标识符：

* **时间**：``phase_t``（当前 phase 已过秒数）、``t``（整个 loop 已过秒数）、
  ``loop_sec``。
* **常量**：``pi``、``tau``、``e``。
* **数学**：``sin``、``cos``、``tan``、``sqrt``、``exp``、``log``、``abs``、
  ``min``、``max``、``clamp(x, lo, hi)``、``lerp(a, b, t)``、
  ``smoothstep(edge0, edge1, x)``。

其他任何东西都会在 parse 时抛 ``ExpressionError``——typo 会变成测试失败，
而不是 frame 200 的时候才悄悄出现 NaN。

骨头别名
^^^^^^^^

不同的 rig 给同一个解剖骨头取不同名字：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 惯例
     - 骨头名示例
   * - VRoid / VRM
     - ``J_Bip_C_Head``、``J_Bip_L_UpperArm``、``J_Bip_L_UpperLeg``
   * - HoYoverse FBX
     - ``Head_M_055``、``Shoulder_L_0183``、``Hip_L_02``
   * - MMD PMX / PMD
     - ``头``、``左腕``、``左足``
   * - Mixamo
     - ``mixamorig:Head``、``mixamorig:LeftArm``

``posecascade.animation.bone_aliasing.detect_humanoid_aliases`` 在 import 时跑，
产出一个 ``{canonical → Node}`` 映射。动画文档引用 canonical name
（``head``、``upper_arm_L``、``upper_leg_L``），这样同一份 JSON 在任何对齐到
那些骨头的 rig 都能播放。可选的 ``rig.body_bones`` 区段可以覆盖自动检测。

布料
^^^^

声明式布料把 PBD 模拟挂到指定的 mesh 上：

.. code-block:: json

   {
     "cloth": [{
       "mesh_node": "skirt",
       "track_bone": "hip",
       "anchor_top_verts": true,
       "stiffness": 0.85,
       "bend_stiffness": 0.20
     }],
     "colliders": [
       { "kind": "sphere",  "follow_bone": "Hip_L_02", "radius": 0.06 },
       { "kind": "capsule", "follow_bone": "Knee_L_04",
         "radius": 0.05, "height": 0.20 }
     ],
     "wind": { "speed": 0.40, "direction": [0, 0, -1] }
   }

``track_bone`` 把顶排锚点顶点粘在会动的骨头上，所以即使角色在地板上走，
布料也会跟着角色走。Collider 是 bone-follow 的 capsule / sphere——布料求解器
每步都会 sweep 每条 edge 对 collider。

完整 schema 见 :doc:`/declarative_animation`。

继承样板（``extends``）
^^^^^^^^^^^^^^^^^^^^^^^

profile JSON 可以把 rig / ground / physics_chains / wind / colliders /
collision_deform_meshes 写一次,所有针对同一角色的动画文件就只引用
这个 profile：

.. code-block:: json

   {
     "schema_version": 1,
     "extends": "_herta_profile.json",
     "name": "my_anim",
     "loop_sec": 4.0,
     "phases": [
       { "name": "do_thing", "duration_sec": 4.0, "pose": "rest_arms" }
     ]
   }

``extends`` 路径相对于文件所在目录,已做 path-traversal 安全检查。
合并是 top-level shallow merge —— 子文件的每个 key 整段替换父文件。
例外是 ``pose_library`` 和 ``hand_library``,这两个会 per-preset 合并。
``phases`` 永远不继承。

随附 ``_herta_profile.json`` 是现成示例—— ``idle.json``、``walk.json``、
``climb_stairs.json``、``showcase.json`` 都从它继承。

简写语法
^^^^^^^^

三种数组简写：

* ``[from, to]`` 取代 ``{"kind": "linear", "from": …, "to": …}``。
* ``[x, y, z]`` 取代 ``{"x": …, "y": …, "z": …}``（``body.translation``）。
  每个元素本身可以是 value curve,所以 ``[0, 0, [0.0, -2.0]]`` 表示
  Z 轴 linear 0 → -2,X / Y 维持常数。
* ``bones`` 区块里 ``x`` / ``y`` / ``z`` 取代 ``x_rad`` / ``y_rad`` / ``z_rad``。

同一根骨头的同一轴同时写长 / 短形（``{"x": 0.5, "x_rad": 0.5}``）会
在 parse 时报错。

----

编辑器内动画编辑器
------------------

PoseCascade 内置一个 in-editor 动画编辑界面：两个共用同一份
in-memory document 的右侧 dock,加上一个共用 undo / redo command stack。
作者可以随时切换“拖卡片”与“打 JSON”两种工作流。

``File → Open Script…`` 打开任何 ``.json`` 动画—— 两个 dock 都会同步
填入。``View → Animation JSON`` 与 ``View → Phase blocks`` 可独立显隐。

**JSON dock** 是个 code editor,per-line 语法着色（key / string /
number / literal / punct）、行号 gutter（parse error 那一行画红色）、
Format 按钮（通过 ``json.dumps`` pretty-print）、dirty indicator
（dock title 带 ``*`` 直到 Save）、Reload 按钮（直接把脚本 re-attach
到 runtime,不必重开窗口）。

**Phase blocks dock** 上面一条横向时间轴（每个 phase 一条 bar,宽度
正比于 ``duration_sec``,点选 / 拖拽重排 / 拖右边缘调 duration）,
下面一列纵向卡片摘要每个 phase。选中卡片会展开 inline 表单,覆盖所有
常用字段：

* **Basic** —— name、duration、blend in/out、pose preset、hand L/R、
  body yaw、body lean X。
* **Gait** —— kind picker（none / walking / stride）,不同 kind 显隐
  对应字段。
* **Body translation** —— XYZ value curves 或 stair block。
* **Bones** —— bone × (x / y / z) curve cell 的表格;点选 cell 会
  弹出 ``CurveEditor``,11 种 curve kind 都支持。
* **Morphs** —— 同样的表格,key 是 morph 名称。

Ctrl+Z / Ctrl+Y 走共用 command stack,所以 undo 在两个 dock 都通。
JSON 编辑器内置的 per-keystroke undo 被关闭;snapshot 一个 typing
session 才取一次,UI 动作各取一次。

完整介绍见 :doc:`/animation_editor`。

----

沙箱 Python 脚本
----------------

``--script`` 传的扩展名是 ``.py`` 时，PoseCascade 通过
``posecascade/scripting/sandbox.py`` 载入脚本。沙箱会：

1. 读取脚本源码（会检查 path-traversal，相对 project root）。
2. 建立受限的 ``globals`` dict，只放 curated API 对象
   （``scene``、``nodes``、``time``、``input``、``math``、``vec3``、``quat``、
   ``lerp``、``clamp``、``noise``）和最小的 builtins 白名单
   （``len``、``range``、``min``、``max``、``abs``、``round``、``enumerate``、
   ``zip``、``print`` → 路由到 logger）。
3. Compile + exec 脚本。
4. 从 globals 拉出用户的 ``update(dt)`` / ``start()`` / ``on_event(...)``
   并存到 script host。

每个 per-frame call 都包在 try/except 里：用户抛的异常会被 log 成
``ScriptRuntimeError``，出错的脚本会被停掉——一个坏脚本永远不会冻住 timeline。

拿不到的东西
^^^^^^^^^^^^

* ``open``、``os``、``sys``、``subprocess``——拿不到文件系统 / 网络。
  要 texture 的话请走引擎的 asset cache。
* ``eval``、``exec``、``compile``、``__import__``——沙箱 loader 是整个 codebase
  **唯一** 的 exec 调用。
* ``__builtins__`` 整个被换掉；用 ``__class__.__mro__`` 之类的小技巧也漏不出来。
* 任何 Qt / GL handle。所有 mutation 都要走 ``scene.find(name)`` →
  ``node.translate/rotate/scale``；渲染线程下一帧会接住变更。

----

渲染管线
--------

前向渲染器每帧按固定顺序跑 **六个 pass**：

1. **Depth-map 阴影 pass**——从主光的视角把 scene render 到一张 depth FBO。
   驱动 self-shadow PCF。
2. **Scene pass**——实际打光 + 着色的 scene。Toon ramp、sphere-map composite、
   inverted-hull outline、可选的 DQS skinning。
3. **Ground pass**——程序化棋盘地板，有 depth，跟渐变天空 blend。
4. **Projected ground shadow**——地面平面上的 quad，把角色剪影投射成柔和 drop shadow。
5. **Selection overlay**——用对比色把选中的 top-level holder 再描一次边。
6. **后处理 chain**——AutoLuminous bloom + MMD tone curve + sRGB-aware 输出。

每个 pass 都有开关（``set_ground_enabled``、``set_self_shadow_enabled``、
``set_projected_shadow_enabled``、``set_selected_holder``），这样 smoke test
与 headless render 可以选择关掉某些 pass 而不影响其他 pass 的像素保真度。

完整拆解——pass 顺序、shader 文件、light-space 数学、texture unit、
MMD-fluence gap——在 :doc:`/rendering_pipeline`。

MMD-fluence 开关
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 功能
     - 做什么
   * - Toon ramp
     - 1-D toon ramp 贴图以 NEAREST + clamp 采样 → 清晰的色块边界，
       没有 linear-filter 的 smear。
   * - Sphere map（mul / add）
     - 把 env-mapped sphere 贴图做 per-pixel composite。PMX 材质
       每个 mat 都带 mode（``MUL`` / ``ADD`` / ``SUB``）。
   * - Inverted-hull outline
     - 第二次 draw pass，cull 正面，顶点沿法线方向 push ``outline_width``——
       经典 MMD 剪影描边。
   * - AutoLuminous bloom
     - per-pixel emission threshold + box-blur + additive composite。
       由 PMX 材质的 ``emission`` channel 驱动。
   * - MMD tone curve
     - 在后处理 chain 最后应用的 hue / saturation / value 重映射。
       特别调过匹配 MikuMikuDance 预设值。
   * - DQS（对偶四元数蒙皮）
     - 可选的 skinning 替代方案（替代 LBS）。极限扭转时保持关节体积——
       手肘不会出现「糖果包装」捏缩。

----

布料求解器
----------

PBD（position-based dynamics），含结构约束 + bend 约束、球体 / 胶囊 collider。
Solver 写了两份：

* **Python orchestrator** (``posecascade/animation/cloth.py``)——
  构建 constraint topology、对 collider 做 broad-phase culling、整合外力
  （重力 + 风）。
* **Cython kernel** (``posecascade/animation/_cloth_kernels.pyx``)——
  per-vertex / per-constraint 的热内循环。由 ``setup.py build_ext --inplace``
  就地构建。

Cython 扩展没编译的话，kernel 会透明退回 NumPy fallback。一样的 API，
速度约 9× 慢。

另外有一条 **GPU compute 路径**,专处理 ``passive_skin_deform`` 布料
（大型角色 mesh 需要 LBS + collider push 但不需要完整 PBD 的场景）。
当前 OpenGL context 为 4.3 以上时引擎自动切到这条快速路径,否则透明
退回 CPU LBS,作者侧没任何感觉。回退条件与哪些平台默认走 GPU 路径
的说明,见 :doc:`/rendering_pipeline`。

----

MCP 服务器
----------

PoseCascade 内置一个 `Model Context Protocol
<https://modelcontextprotocol.io/>`_ 服务器，让任何 MCP-aware 的 LLM agent
不用通过桌面 UI 就能驱动引擎。服务器是 headless 的——完全不会碰 Qt 或 GL
context——所以可以在 subprocess 走 stdio 干净地跑。

带 ``ai`` extra 安装：

.. code-block:: bash

   pip install -e .[ai]

那会拉进 ``mcp`` 与 ``jsonschema``，并安装 ``posecascade-mcp`` console script。

Repo 内附的 ``.mcp.json`` 是项目层级配置，Claude Code（与其他 MCP-aware 客户端）
在 venv 在 PATH 上的时候会自动 pick up：

.. code-block:: json

   {
     "$schema": "https://modelcontextprotocol.io/schema/server-config.json",
     "mcpServers": {
       "posecascade": {
         "command": "posecascade-mcp",
         "args": [],
         "env": {}
       }
     }
   }

提供的工具
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - 工具
     - 用途
   * - ``list_animations``
     - 列出 ``examples/scripts/`` 下所有 ``.json`` 声明式动画，
       带 ``name``、``loop_sec`` 和 phase 数。
   * - ``read_animation``
     - 返回某个文件的原始 JSON 文本。
   * - ``validate_animation``
     - JSON-Schema 检查加上 runtime parser 检查。``content`` (行内)
       或 ``path`` (项目相对路径) 择一。
   * - ``inspect_model``
     - 导入任何支持的模型，返回结构摘要
       （mesh / texture / skin / node / vertex / triangle 数量、
       前 20 个骨头名、world AABB）。
   * - ``cloth_benchmark``
     - 建一个合成 grid、加上重力 + 球体 collider、跑 N 步求解器。
       返回 ms/step（三次中最佳）、``frame_section`` 拆解，以及
       ``native_kernel`` flag。

完整 API（function signature、路径安全、schema 细节）见 :doc:`/mcp`。

----

故障排查
--------

GL context 建立失败
^^^^^^^^^^^^^^^^^^^

PoseCascade 需要 **OpenGL 3.3 core profile** 或更新。在没有最新驱动的老 Intel
iGPU 上，Qt 可能 fall back 到 OpenGL 1.4 然后在 shader compile 时 crash。
解法：

* 更新显卡驱动（Windows 上 Intel HD 4000+ 支持 GL 3.3）。
* Headless render 可以用 ``QT_QPA_PLATFORM=offscreen`` 强制软件绘图，
  或在 Debian 系 Linux 上装 Mesa software rasteriser（``libgl1-mesa-glx``）。
* 只需要 MCP 服务器的话，``ai`` extra 在任何 CPU 上都能跑——
  不会建 GL context。

导入的模型看不见 / 只剩一根骨头
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最常见的情况是模型里有单独的 mesh（武器、prop、stage piece）触发了 importer
警告。在 Blender 打开文件、移除那块 mesh、重新导出。随附的 March 7th GLB
就走过这个流程——两个武器 mesh 在进 repo 之前被剥掉了。

如果走 PMX 路径、toon 着色看起来糊掉了，确认 toon ramp 贴图
（``toon01.bmp`` … ``toon10.bmp``）在 ``.pmx`` 文件同一个文件夹。PMX 材质
用相对路径引用 toon 贴图。

----

GUI 界面 + 多语言
-----------------

编辑器用户可见的所有界面——每个菜单、每个 dock、工具栏、状态栏、
导出对话框、3D 视窗的鼠标操作——有独立的完整说明在 :doc:`/gui`。
那里涵盖而本文未重复的主题：

* 每个 dock 的行为（Outliner、Slots、Inspector、Effects、Timeline、
  Tracks、Animation JSON、Phase blocks）。
* 每帧循环、暂停标志、出错脚本如何被隔离以避免冻住 timeline。
* DPI 感知尺寸与用 ``QFontMetrics`` 推导出的组件最小宽高
  （取代旧版的像素硬编值）。

GUI 默认搭载三个语种（英文、繁体中文、简体中文），可在运行中从
``设置 → 语言`` 切换。完整 i18n 参考——目录文件格式、语种解析顺序、
如何以单一文件新增语言、翻译者必须保留的 placeholder 规则、
搭配的响应式尺寸策略——在 :doc:`/internationalization`。

要强制单次启动使用特定语种：

.. code-block:: bash

   POSECASCADE_LANG=zh-CN python -m posecascade

``POSECASCADE_LANG`` 同时覆盖 ``设置 → 语言`` 持久化的选择
与操作系统报告的语种。

----

许可 + 致谢
-----------

Codebase 采 **MIT-style** 许可；详细条款见 ``LICENSE``。

随附素材各有自己的许可：

* ``examples/assets/herta/herta.glb``——CC-BY 4.0
  （上传者 *X9_YT* on Sketchfab）。角色「The Herta」© HoYoverse，
  模型依其 **Fan Content Guidelines** 使用：禁商业利用、保留 attribution、
  不可贬损使用。完整 attribution 在
  ``examples/assets/herta/NOTICE.md``。
* ``examples/assets/march7th/march7th.pmx``——CC-BY 4.0
  （上传者 *Gregman* on Sketchfab）。角色「March 7th」© HoYoverse，
  适用同一份 Fan Content Guidelines。完整 attribution 在
  ``examples/assets/march7th/NOTICE.md``。
* 默认天空 / 地面贴图——public domain（CC0）。

如果你 fork PoseCascade 做商业产品，请把 ``examples/assets/herta/``
与 ``examples/assets/march7th/`` 换成你有权使用的模型。引擎本身没有
MMD / HoYoverse 专属的代码路径——任何 PBR 蒙皮的人形角色都能套用
同一层别名。
