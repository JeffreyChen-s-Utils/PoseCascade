PoseCascade 使用手冊
====================

一個以 PySide6 + OpenGL 打造的桌面引擎，用來匯入 3D 模型並以沙箱化腳本驅動它們。
視覺目標是 **MMD**：卡通著色搭配清晰的 inverted-hull 描邊、PMX 材質帶 sphere map、
VMD 風格動畫曲線、IK + 腳掌貼地、morph target，以及可即時擺動數件布料的 PBD 布料求解器。

.. contents:: 目錄
   :depth: 2
   :local:

----

安裝
----

PoseCascade 鎖定 Python **3.14**，支援 Windows、macOS、Linux。建議使用專案內的
virtualenv，這樣 editable install 編譯 Cython 布料 kernel 時能對到同一個 interpreter：

.. code-block:: bash

   git clone https://github.com/JeffreyChen-s-Utils/PoseCascade.git
   cd PoseCascade
   python -m venv .venv
   .venv\Scripts\Activate.ps1            # Windows PowerShell
   # source .venv/bin/activate           # Linux / macOS

   pip install -e .[dev,ai]

``dev`` extra 會拉進 ``pytest``、``pytest-qt``、``ruff``、``bandit`` 與
``scikit-image``（golden-image SSIM 容差用）。``ai`` extra 會安裝可選的 ``mcp``
伺服器進入點。

.. note::
   editable install 會就地編譯 ``posecascade/animation/_cloth_kernels.pyx``。
   Windows 需要 **Microsoft Build Tools**；Linux / macOS 需要 gcc 或 clang。
   沒有 C 編譯器的話，``pip`` 會印警告，引擎會透明退回 NumPy fallback 路徑——
   速度比較慢，但功能上完全一致。

可選相依
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Extra
     - 解鎖什麼
   * - ``[ai]``
     - ``mcp`` + ``jsonschema``——安裝 ``posecascade-mcp`` CLI，
       供 Claude Code、Claude Desktop、Cursor … 使用。
   * - ``[fbx]``
     - FBX importer 外掛（獨立 runtime 因為 FBX SDK 是 Autodesk 授權且體積大）。
   * - ``[usd]``
     - Universal Scene Description 外掛（Pixar 的 USD bindings）。
   * - ``[collada]``
     - ``.dae`` importer（XML-based；用 ``defusedxml`` 避免 XXE 攻擊）。

----

執行範例
--------

隨附範例放在 ``examples/``。分三類：

互動式 viewport（聲明式 JSON）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最簡單看到角色動起來的方式。``--scene`` 傳模型，``--script`` 傳 ``.json``
動畫文件：

.. code-block:: bash

   # 原地走路 + 擺手 —— 4 秒迴圈
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/walk.json

   # 30 秒展示輪播 —— idle → turntable → wave → V-pose → bow
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/showcase.json

   # 上下樓梯（聲明式 phases）
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/climb_stairs.json

   # 最小呼吸 idle 迴圈
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/idle.json

   # 四足「狗爬式」——先擺到雙手雙膝著地，再向前爬行
   python -m posecascade --scene examples/assets/herta/herta.glb \
                          --script examples/scripts/dog_crawl.json

互動式 viewport（沙箱 Python）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

把副檔名換成 ``.py`` 就會載入沙箱 Python 腳本而不是 JSON 文件。
適合 per-frame 邏輯不適合塞進 phases 的情境（例如純布料展示、頭髮物理示範）：

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

獨立腳本，不開 Qt 視窗——它們開一個 offscreen GL context、渲染單張 frame
（或 N 張的 strip），然後把結果寫到磁碟。適合當整個視覺管線的快速 smoke check：

.. code-block:: bash

   # MMD 風格 hero frame，每個 fluence toggle 都開
   python examples/mmd_demo.py

   # PMX-native 路徑（per-mesh MMDMaterial + sphere texture）
   # --frames N 會輸出 N 張的動畫條
   python examples/march7th_pmx_demo.py --frames 8

   # 並排比較渲染
   python examples/compare_bloom.py     # AutoLuminous bloom 關 vs 開
   python examples/compare_dqs.py       # LBS 糖果包裝 vs DQS
   python examples/compare_lights.py    # 只開主光 vs HighDef 多光
   python examples/compare_tone.py      # 純 sRGB vs + mmd_tone

   # 360° 轉場（輸出 spin.mp4）
   python examples/spin.py

隨附角色
^^^^^^^^

所有隨附範例都使用 ``examples/assets/herta/herta.glb``
（《崩壞：星穹鐵道》的「The Herta」——354 個 joint，連身裙綁入身體 mesh）。
MMD 演示路徑用另一個 PMX 資產：

.. code-block:: bash

   python -m posecascade --scene examples/assets/march7th/march7th.pmx

兩個素材都是第三方 CC-BY 4.0。glTF 上傳者為 Sketchfab 上的 X9_YT
（角色「The Herta」© HoYoverse，依其 Fan Content Guidelines 使用）；
PMX 上傳者為 Sketchfab 上的 Gregman（角色「March 7th」© HoYoverse）。
完整聲明分別見 ``examples/assets/herta/NOTICE.md`` 與
``examples/assets/march7th/NOTICE.md``。

----

編寫聲明式動畫
--------------

聲明式動畫是放在 ``examples/scripts/`` 下的 JSON 文件，把角色綁進一連串的 **phases**。
每個 phase 指定持續時間、身體做什麼（translation、yaw、lean）、肢體做什麼
（gait、IK target、pose preset），以及 morph 做什麼（smile、blink、mouth-A）。
Runtime 會在 phases 之間 cross-fade，跑到結尾自動 loop，把每幀的 pose 餵給渲染器。

文件結構
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

頂層 key：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - 用途
   * - ``schema_version``
     - 永遠是 ``1``。未來不向下相容的變更會 bump。
   * - ``name``
     - log 與 MCP ``list_animations`` 工具用的穩定識別碼。
   * - ``loop_sec``
     - 一個 loop 的總長度。必須等於所有 phase 的 ``duration_sec`` 總和
       （parser 會檢查）。
   * - ``rig``
     - 骨頭命名別名——見下面的 *骨頭別名*。
   * - ``ground``
     - 可選的平地面（Y 高度），給腳掌 IK 夾鉗用。
   * - ``phases``
     - 有順序的清單。每個項目跑 ``duration_sec`` 秒。
   * - ``hide``
     - 可選清單，列出開始時要從 scene 取下的 node 名稱
       （動畫不想要的 prop、燈光、樓梯）。
   * - ``cloth`` / ``colliders``
     - 可選的聲明式布料設定——見下面的 *布料*。
   * - ``pose_library``
     - 可選的文件內 pose preset，會覆蓋同名 built-in。

Phase 內容
^^^^^^^^^^

每個 phase 可以宣告下列一個或多個欄位：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 欄位
     - 驅動什麼
   * - ``body``
     - 整個角色的 yaw、translation、lean。值是 expressions，每幀求值
       （``"phase_t"`` 是 phase 內已過的秒數）。
   * - ``gait``
     - 程序化的走 / 跑 cycle。用解析公式設定腿 + 手臂角度。
       ``kind`` 可以是 ``"walking"``、``"running"``、``"idle"``。
   * - ``ik``
     - 腳掌 IK 目標。配合 ``ground`` 夾鉗能讓腳掌一直在地板上。
   * - ``poses``
     - 命名 preset 混合（``reach_R_soft``、``wave_L``、``T_pose`` …）。
       Built-in preset 在 ``posecascade/scripting/pose_library.py``。
   * - ``morphs``
     - per-morph 權重，支援 expression
       （``"smile": "0.5 + 0.5*sin(phase_t * pi)"``）。
   * - ``cross_fade_sec``
     - 從前一個 phase 緩進到這個 phase 的秒數。Runtime 會內插 pose + morph 權重。

Expression DSL
^^^^^^^^^^^^^^

數值欄位接受 literal 數值或 string expression。DSL 是 Python 的安全子集——
沒有 ``import``、沒有 attribute access、沒有 function definition。可用識別子：

* **時間**：``phase_t``（當前 phase 已過秒數）、``t``（整個 loop 已過秒數）、
  ``loop_sec``。
* **常數**：``pi``、``tau``、``e``。
* **數學**：``sin``、``cos``、``tan``、``sqrt``、``exp``、``log``、``abs``、
  ``min``、``max``、``clamp(x, lo, hi)``、``lerp(a, b, t)``、
  ``smoothstep(edge0, edge1, x)``。

其他任何東西都會在 parse 時拋 ``ExpressionError``——typo 會變成測試失敗，
而不是 frame 200 的時候才悄悄出現 NaN。

骨頭別名
^^^^^^^^

不同的 rig 給同一個解剖骨頭取不同名字：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 慣例
     - 骨頭名範例
   * - VRoid / VRM
     - ``J_Bip_C_Head``、``J_Bip_L_UpperArm``、``J_Bip_L_UpperLeg``
   * - HoYoverse FBX
     - ``Head_M_055``、``Shoulder_L_0183``、``Hip_L_02``
   * - MMD PMX / PMD
     - ``頭``、``左腕``、``左足``
   * - Mixamo
     - ``mixamorig:Head``、``mixamorig:LeftArm``

``posecascade.animation.bone_aliasing.detect_humanoid_aliases`` 在 import 時跑，
產出一個 ``{canonical → Node}`` 對映。動畫文件參考 canonical name
（``head``、``upper_arm_L``、``upper_leg_L``），這樣同一份 JSON 在任何有對齊到
那些骨頭的 rig 都能播放。可選的 ``rig.body_bones`` 區塊可以覆蓋自動偵測。

布料
^^^^

聲明式布料把 PBD 模擬掛到指定的 mesh 上：

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

``track_bone`` 把頂排錨點頂點黏在會動的骨頭上，所以即使角色在地板上走，
布料也會跟著角色走。Collider 是 bone-follow 的 capsule / sphere——布料求解器
每步都會 sweep 每條 edge 對 collider。

完整 schema 見 :doc:`/declarative_animation`。

繼承樣板（``extends``）
^^^^^^^^^^^^^^^^^^^^^^^

profile JSON 可以把 rig / ground / physics_chains / wind / colliders /
collision_deform_meshes 寫一次，所有針對同一角色的動畫檔就只 reference
這個 profile：

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

``extends`` 路徑相對於檔案所在目錄，已做 path-traversal 安全檢查。
合併是 top-level shallow merge——子檔的每個 key 整段蓋掉父檔。例外是
``pose_library`` 和 ``hand_library``，這兩個會 per-preset 合併。
``phases`` 永遠不繼承。

隨附 ``_herta_profile.json`` 是現成範例——``idle.json``、``walk.json``、
``climb_stairs.json``、``showcase.json`` 都從它繼承。

簡寫語法
^^^^^^^^

三種陣列簡寫：

* ``[from, to]`` 取代 ``{"kind": "linear", "from": …, "to": …}``。
* ``[x, y, z]`` 取代 ``{"x": …, "y": …, "z": …}``（``body.translation``）。
  每個元素本身可以是 value curve，所以 ``[0, 0, [0.0, -2.0]]`` 表示
  Z 軸 linear 0 → -2，X / Y 維持常數。
* ``bones`` 區塊裡 ``x`` / ``y`` / ``z`` 取代 ``x_rad`` / ``y_rad`` / ``z_rad``。

同一根骨頭的同一軸同時寫長 / 短形（``{"x": 0.5, "x_rad": 0.5}``）會
在 parse 時報錯。

----

編輯器內動畫編輯器
------------------

PoseCascade 內建一個 in-editor 動畫編輯介面：兩個共用同一份
in-memory document 的右側 dock，加上一個共用 undo / redo command stack。
作者可以隨時切換「拖卡片」跟「打 JSON」兩種工作流。

``File → Open Script…`` 開啟任何 ``.json`` 動畫——兩個 dock 都會同步
填入。``View → Animation JSON`` 跟 ``View → Phase blocks`` 可獨立顯隱。

**JSON dock** 是個 code editor，per-line 語法上色（key / string /
number / literal / punct）、行號 gutter（parse error 那一行畫紅色）、
Format 按鈕（透過 ``json.dumps`` pretty-print）、dirty indicator
（dock title 帶 ``*`` 直到 Save）、Reload 按鈕（直接把腳本 re-attach
到 runtime，不必重開視窗）。

**Phase blocks dock** 上面一條橫向時間軸（每個 phase 一條 bar，寬度
正比於 ``duration_sec``，點選 / 拖曳重排 / 拖右邊緣調 duration），
下面一列縱向卡片摘要每個 phase。選中卡片會展開 inline 表單，覆蓋所有
常用欄位：

* **Basic** —— name、duration、blend in/out、pose preset、hand L/R、
  body yaw、body lean X。
* **Gait** —— kind picker（none / walking / stride），不同 kind 顯隱
  對應欄位。
* **Body translation** —— XYZ value curves 或 stair block。
* **Bones** —— bone × (x / y / z) curve cell 的表格；點選 cell 會
  彈出 ``CurveEditor``，11 種 curve kind 都支援。
* **Morphs** —— 同樣的表格，key 是 morph 名稱。

Ctrl+Z / Ctrl+Y 走共用 command stack，所以 undo 在兩個 dock 都通。
JSON 編輯器內建的 per-keystroke undo 被關閉；snapshot 一個 typing
session 才取一次，UI 動作各取一次。

完整介紹見 :doc:`/animation_editor`。

----

沙箱 Python 腳本
----------------

``--script`` 傳的副檔名是 ``.py`` 時，PoseCascade 透過
``posecascade/scripting/sandbox.py`` 載入腳本。沙箱會：

1. 讀取腳本原始碼（會檢查 path-traversal，相對 project root）。
2. 建立受限的 ``globals`` dict，只放 curated API 物件
   （``scene``、``nodes``、``time``、``input``、``math``、``vec3``、``quat``、
   ``lerp``、``clamp``、``noise``）和最小的 builtins 白名單
   （``len``、``range``、``min``、``max``、``abs``、``round``、``enumerate``、
   ``zip``、``print`` → 路由到 logger）。
3. Compile + exec 腳本。
4. 從 globals 拉出使用者的 ``update(dt)`` / ``start()`` / ``on_event(...)``
   並存到 script host。

每個 per-frame call 都包在 try/except 裡：使用者拋的例外會被 log 成
``ScriptRuntimeError``，出錯的腳本會被停掉——一個壞腳本永遠不會凍住 timeline。

拿不到的東西
^^^^^^^^^^^^

* ``open``、``os``、``sys``、``subprocess``——拿不到檔案系統 / 網路。
  要 texture 的話請走引擎的 asset cache。
* ``eval``、``exec``、``compile``、``__import__``——沙箱 loader 是整個 codebase
  **唯一** 的 exec 呼叫。
* ``__builtins__`` 整個被換掉；用 ``__class__.__mro__`` 之類的小技巧也漏不出來。
* 任何 Qt / GL handle。所有 mutation 都要走 ``scene.find(name)`` →
  ``node.translate/rotate/scale``；渲染執行緒下一幀會接住變更。

----

渲染管線
--------

前向渲染器每幀依固定順序跑 **六個 pass**：

1. **Depth-map 陰影 pass**——從主光的視角把 scene render 到一張 depth FBO。
   驅動 self-shadow PCF。
2. **Scene pass**——實際打光 + 著色的 scene。Toon ramp、sphere-map composite、
   inverted-hull outline、可選的 DQS skinning。
3. **Ground pass**——程序化棋盤地板，有 depth，跟漸層天空 blend。
4. **Projected ground shadow**——地面平面上的 quad，把角色剪影投射成柔和 drop shadow。
5. **Selection overlay**——用對比色把選中的 top-level holder 再描一次邊。
6. **後處理 chain**——AutoLuminous bloom + MMD tone curve + sRGB-aware 輸出。

每個 pass 都有開關（``set_ground_enabled``、``set_self_shadow_enabled``、
``set_projected_shadow_enabled``、``set_selected_holder``），這樣 smoke test
與 headless render 可以選擇關掉某些 pass 而不影響其他 pass 的像素保真度。

完整拆解——pass 順序、shader 檔案、light-space 數學、texture unit、
MMD-fluence gap——在 :doc:`/rendering_pipeline`。

MMD-fluence 開關
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 功能
     - 做什麼
   * - Toon ramp
     - 1-D toon ramp 貼圖以 NEAREST + clamp 取樣 → 清晰的色塊邊界，
       沒有 linear-filter 的 smear。
   * - Sphere map（mul / add）
     - 把 env-mapped sphere 貼圖做 per-pixel composite。PMX 材質
       每個 mat 都帶 mode（``MUL`` / ``ADD`` / ``SUB``）。
   * - Inverted-hull outline
     - 第二次 draw pass，cull 正面，頂點沿法線方向 push ``outline_width``——
       經典 MMD 剪影描邊。
   * - AutoLuminous bloom
     - per-pixel emission threshold + box-blur + additive composite。
       由 PMX 材質的 ``emission`` channel 驅動。
   * - MMD tone curve
     - 在後處理 chain 最後套用的 hue / saturation / value 重映射。
       特別調過匹配 MikuMikuDance 預設值。
   * - DQS（對偶四元數蒙皮）
     - 可選的 skinning 替代方案（替代 LBS）。極限扭轉時保持關節體積——
       手肘不會出現「糖果包裝」捏縮。

----

布料求解器
----------

PBD（position-based dynamics），含結構約束 + bend 約束、球體 / 膠囊 collider。
Solver 寫了兩份：

* **Python orchestrator** (``posecascade/animation/cloth.py``)——
  建構 constraint topology、對 collider 做 broad-phase culling、整合外力
  （重力 + 風）。
* **Cython kernel** (``posecascade/animation/_cloth_kernels.pyx``)——
  per-vertex / per-constraint 的熱內迴圈。由 ``setup.py build_ext --inplace``
  就地建構。

Cython 擴展沒編譯的話，kernel 會透明退回 NumPy fallback。一樣的 API，
速度約 9× 慢。

另外有一條 **GPU compute 路徑**，專處理 ``passive_skin_deform`` 布料
（大型角色 mesh 需要 LBS + collider push 但不需要完整 PBD 的場景）。
作用中的 OpenGL context 為 4.3 以上時引擎自動切到這條快速路徑，否則
透明退回 CPU LBS，作者那一側沒任何感覺。回退條件與哪些平台預設走 GPU
路徑的說明，見 :doc:`/rendering_pipeline`。

----

MCP 伺服器
----------

PoseCascade 內建一個 `Model Context Protocol
<https://modelcontextprotocol.io/>`_ 伺服器，讓任何 MCP-aware 的 LLM agent
不用透過桌面 UI 就能驅動引擎。伺服器是 headless 的——完全不會碰 Qt 或 GL
context——所以可以在 subprocess 走 stdio 乾淨地跑。

帶 ``ai`` extra 安裝：

.. code-block:: bash

   pip install -e .[ai]

那會拉進 ``mcp`` 與 ``jsonschema``，並安裝 ``posecascade-mcp`` console script。

Repo 內附的 ``.mcp.json`` 是專案層級設定，Claude Code（與其他 MCP-aware 客戶端）
在 venv 在 PATH 上的時候會自動 pick up：

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
     - 列出 ``examples/scripts/`` 底下所有 ``.json`` 聲明式動畫，
       帶 ``name``、``loop_sec`` 和 phase 數。
   * - ``read_animation``
     - 回傳某個檔案的原始 JSON 文字。
   * - ``validate_animation``
     - JSON-Schema 檢查加上 runtime parser 檢查。``content`` (行內)
       或 ``path`` (專案相對路徑) 擇一。
   * - ``inspect_model``
     - 匯入任何支援的模型，回傳結構摘要
       （mesh / texture / skin / node / vertex / triangle 數量、
       前 20 個骨頭名、world AABB）。
   * - ``cloth_benchmark``
     - 建一個合成 grid、加上重力 + 球體 collider、跑 N 步求解器。
       回傳 ms/step（三次中最佳）、``frame_section`` 拆解，以及
       ``native_kernel`` flag。

完整 API（function signature、路徑安全、schema 細節）見 :doc:`/mcp`。

----

疑難排解
--------

GL context 建立失敗
^^^^^^^^^^^^^^^^^^^

PoseCascade 需要 **OpenGL 3.3 core profile** 或更新。在沒有最新驅動的老 Intel
iGPU 上，Qt 可能 fall back 到 OpenGL 1.4 然後在 shader compile 時 crash。
解法：

* 更新顯卡驅動（Windows 上 Intel HD 4000+ 支援 GL 3.3）。
* Headless render 可以用 ``QT_QPA_PLATFORM=offscreen`` 強制軟體繪圖，
  或在 Debian 系 Linux 上裝 Mesa software rasteriser（``libgl1-mesa-glx``）。
* 只需要 MCP 伺服器的話，``ai`` extra 在任何 CPU 上都能跑——
  不會建 GL context。

匯入的模型看不見 / 只剩一根骨頭
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最常見的情況是模型裡有單獨的 mesh（武器、prop、stage piece）觸發了 importer
警告。在 Blender 開啟檔案、移除那塊 mesh、重新匯出。隨附的 March 7th GLB
就走過這個流程——兩個武器 mesh 在進 repo 之前被剝掉了。

如果走 PMX 路徑、toon 著色看起來糊掉了，確認 toon ramp 貼圖
（``toon01.bmp`` … ``toon10.bmp``）在 ``.pmx`` 檔同一個資料夾。PMX 材質
用相對路徑引用 toon 貼圖。

----

GUI 介面 + 多語言
-----------------

編輯器使用者可見的所有介面——每個選單、每個 dock、工具列、狀態列、
匯出對話框、3D 視窗的滑鼠操作——有獨立的完整說明在 :doc:`/gui`。
那裡涵蓋而本文未重複的主題：

* 每個 dock 的行為（Outliner、Slots、Inspector、Effects、Timeline、
  Tracks、Animation JSON、Phase blocks）。
* 每幀迴圈、暫停旗標、出錯腳本如何被隔離以避免凍住 timeline。
* DPI 感知尺寸與用 ``QFontMetrics`` 推導出的元件最小寬高
  （取代舊版的像素硬編值）。

GUI 預設搭載三個語系（英文、繁體中文、簡體中文），可在執行中從
``設定 → 語言`` 切換。完整 i18n 參考——目錄檔案格式、語系決定順序、
如何以單一檔案新增語言、翻譯者必須保留的 placeholder 規則、
搭配的響應式尺寸策略——在 :doc:`/internationalization`。

要強制單次啟動使用特定語系：

.. code-block:: bash

   POSECASCADE_LANG=zh-TW python -m posecascade

``POSECASCADE_LANG`` 同時覆寫 ``設定 → 語言`` 持久化的選擇
與作業系統回報的語系。

----

授權 + 致謝
-----------

Codebase 採 **MIT-style** 授權；詳細條款見 ``LICENSE``。

隨附素材各有自己的授權：

* ``examples/assets/herta/herta.glb``——CC-BY 4.0
  （上傳者 *X9_YT* on Sketchfab）。角色「The Herta」© HoYoverse，
  模型依其 **Fan Content Guidelines** 使用：禁商業利用、保留 attribution、
  不可貶損使用。完整 attribution 在
  ``examples/assets/herta/NOTICE.md``。
* ``examples/assets/march7th/march7th.pmx``——CC-BY 4.0
  （上傳者 *Gregman* on Sketchfab）。角色「March 7th」© HoYoverse，
  適用同一份 Fan Content Guidelines。完整 attribution 在
  ``examples/assets/march7th/NOTICE.md``。
* 預設天空 / 地面貼圖——public domain（CC0）。

如果你 fork PoseCascade 做商業產品，請把 ``examples/assets/herta/``
與 ``examples/assets/march7th/`` 換成你有權使用的模型。引擎本身沒有
MMD / HoYoverse 專屬的程式碼路徑——任何 PBR 蒙皮的人形角色都能套用
同一層別名。
