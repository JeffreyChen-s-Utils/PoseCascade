# PoseCascade

> **語言**：[English](README.md) · **繁體中文** · [简体中文](README.zh-CN.md)
> **文件**：[Read the Docs 原始碼](docs/)（Sphinx）

一個以 PySide6 + OpenGL 打造的桌面引擎，用來匯入 3D 模型並以沙箱化腳本驅動它們。
視覺目標是 MMD 風格：卡通著色搭配清晰的 inverted-hull 描邊、PMX 材質帶 sphere map、
VMD 風格動畫曲線、IK + 腳掌貼地、morph target，以及可即時擺動數件布料的 PBD 布料求解器。

## 功能特色

- **多格式匯入**：glTF / GLB、OBJ + MTL、FBX、STL、PLY、USD / USDZ、COLLADA，
  以及 MMD 三件套（PMX 模型 + VMD 動作 + VPD 姿態）。每個格式都掛在自己的外掛 adapter
  後面，新增格式不會動到渲染器。
- **MMD 味的前向渲染器**：toon ramp（NEAREST + clamp 取得乾淨色塊邊界）、sphere-map
  合成、inverted-hull 描邊、程序化棋盤地面、投影地面陰影、深度貼圖 PCF-softened
  自我陰影、sRGB-aware 輸出、漸層天空 pass、多光源 HighDef 配置（1 主光 + 最多 3 副光）、
  可選的對偶四元數蒙皮（保持關節體積）、預設 AutoLuminous bloom、MMD tone-curve 後處理、
  程序化舞台抽象（地板 + 背牆 + 側牆），以及一個 selection-overlay pass，會用對比色把
  選中的 top-level holder 再描一次邊。完整 pass 順序與每個 pass 的開關見
  [`docs/rendering_pipeline.md`](docs/rendering_pipeline.md)。
- **VMD 驅動動畫**：per-bone / per-morph / per-camera 軌道，使用 MMD 慣用的四控制點
  貝茲插值、IK 求解器、腳掌貼地器、slot 之間的 external-parent 綁定、display-frame
  群組以及 physics 鏈。
- **布料求解器**：position-based dynamics 含結構約束 + bend 約束、球體 / 膠囊碰撞體
  （含連續碰撞 sweep），以及一個 Cython kernel，在 480-vert 裙子上把每步成本壓到
  **0.35 ms** 左右——比純 NumPy fallback 快約 9×。若編譯擴展未建構，kernel 會透明
  地退回 NumPy 路徑。
- **聲明式動畫 runtime**：用 JSON 文件而不是 Python 程式碼來驅動角色——phases、
  gaits、body trajectories、morph 時間線、行內 expression DSL，加上 `extends`
  繼承機制與陣列簡寫（`[from, to]` 曲線、`[x, y, z]` 平移、bones 的 `x`/`y`/`z`
  軸別名），把典型動畫檔長度縮到原本長寫法的 1/3。詳見
  [`docs/declarative_animation.md`](docs/declarative_animation.md)。
- **編輯器內動畫編輯器**（新）：兩個共用同一份 in-memory document 的右側 dock——
  JSON 編輯器具備語法上色、行號 gutter、parse error 行內標記、Format 按鈕、
  dirty indicator；Phase 方塊 dock 則有橫向時間軸（拖曳重排 + 拖邊緣調 duration）、
  縱向卡片列表，加上 inline 表單覆蓋所有常用欄位（name / duration / blends /
  pose / gait / body translation / bones / morphs）。兩邊都接 Ctrl+Z / Ctrl+Y
  undo/redo。詳見 [`docs/animation_editor.md`](docs/animation_editor.md)。
- **GPU compute 蒙皮**（新）：用 OpenGL 4.3 compute shader 把 `passive_skin_deform`
  布料的 LBS + 碰撞推開 + world-to-local 都搬到 GPU，直接寫進 mesh 的 position
  與 normal VBO。在 30k-vert 身體 mesh 上把每幀 cloth + apply_cloth 從 ~9 ms 降到
  0.05 ms 以下；GL 版本低於 4.3 或 compute 編譯失敗時透明退回 CPU LBS 路徑。
- **沙箱化 Python 腳本**：在受限的命名空間裡執行（拿不到 `open`、`os`、`subprocess`、
  `__import__` …），對外只提供 `scene`、`nodes`、`time`、`input`，以及一層整理過的
  數學 API，讓使用者擺姿勢、打 keyframe、做動畫，不必碰引擎內部。
- **MCP 伺服器**：一個 Model Context Protocol 伺服器，讓任何支援 MCP 的 LLM agent
  都能驅動引擎——列出並驗證聲明式動畫腳本、檢查模型、跑布料 benchmark。詳見
  [`docs/mcp.md`](docs/mcp.md)。

## 快速開始

```bash
# Clone + 建立虛擬環境
git clone https://github.com/JeffreyChen-s-Utils/PoseCascade.git
cd PoseCascade
python -m venv .venv
.venv\Scripts\activate.ps1            # Windows PowerShell
# source .venv/bin/activate           # Linux / macOS

# 帶 AI extras 安裝，這樣連 MCP 伺服器一起裝起來
pip install -e .[dev,ai]
```

editable install 會就地編譯 Cython 布料 kernel。沒有 C 編譯器
（Windows 上的 Microsoft Build Tools、Linux / macOS 上的 gcc / clang）時，
安裝會印警告，引擎透明退回 NumPy 路徑。

針對隨附範例啟動編輯器：

```bash
# 經典 3D 模型展示輪播（30 秒）：intro idle → 360° turntable →
# 原地走路 → 揮手 → V-pose → hip-pop → 鞠躬 → 回到 neutral。
python -m posecascade --scene examples/assets/herta/herta.glb \
                       --script examples/scripts/showcase.json

# 原地行走 + 擺手 4 秒迴圈。
python -m posecascade --scene examples/assets/herta/herta.glb \
                       --script examples/scripts/walk.json

# 最小呼吸 idle 4 秒迴圈。
python -m posecascade --scene examples/assets/herta/herta.glb \
                       --script examples/scripts/idle.json
```

或者 headless 直接渲染 MMD 風格 hero frame——不開編輯器視窗，
適合當整套視覺管線的 smoke check：

```bash
# 載入 herta.glb（glTF），在 force_toon 路徑上開所有 MMD-fluence
# 開關，輸出 mmd_demo.png 到腳本旁邊。
python examples/mmd_demo.py

# 載入隨附的 March 7th PMX，走渲染器的 PMX-native 路徑
# （per-mesh MMDMaterial + sphere texture + 直接從檔案讀的 edge flag）。
# 這個素材是第三方資源，授權說明見 examples/assets/march7th/NOTICE.md。
# 輸出 march7th_pmx_demo.png 到腳本旁邊。
python examples/march7th_pmx_demo.py

# 也可以透過互動式編輯器執行：
python -m posecascade --scene examples/assets/march7th/march7th.pmx
```

一組並排對比腳本展示每個 MMD-fluence 功能 vs baseline——
每個都會輸出標好標籤的 PNG，讓視覺差異可重現：

```bash
python examples/compare_bloom.py    # bloom 關閉 vs 套用 AutoLuminous
python examples/compare_tone.py     # 純 sRGB vs + mmd_tone
python examples/compare_dqs.py      # LBS 糖果包裝 vs DQS 極限扭轉
python examples/compare_lights.py   # 只開主光 vs + HighDef rim + fill
```

## 文件

[`docs/`](docs/) 下的 Sphinx 樹涵蓋：

- [`docs/rendering_pipeline.md`](docs/rendering_pipeline.md) —— 每個 render
  pass 做什麼、開關怎麼按。
- [`docs/declarative_animation.md`](docs/declarative_animation.md) ——
  撰寫 JSON 動畫：phases、gaits、curves、expression DSL、`extends`
  profile 繼承,以及陣列簡寫。
- [`docs/animation_editor.md`](docs/animation_editor.md) —— 編輯器內
  JSON dock + Phase blocks dock(時間軸 + 曲線編輯器 + undo/redo)
  的使用說明。
- [`docs/mcp.md`](docs/mcp.md) —— Model Context Protocol 伺服器的
  工具、參數、回傳格式。
- [`docs/packaging_pyinstaller.md`](docs/packaging_pyinstaller.md)
  —— 用 PyInstaller 把 PoseCascade 包成獨立執行檔。
- [`docs/packaging_nuitka.md`](docs/packaging_nuitka.md) —— 同樣的
  事改用 Nuitka(編譯成原生二進位,體積更小 / 啟動更快,但 build 較久)。

貢獻者 / 維護者文件(開發流程、CI、release pipeline、效能數字)在
[`DEVELOPMENT.md`](DEVELOPMENT.md)。

## 授權

專案的 MIT-style 條款見 [`LICENSE`](LICENSE)。隨附素材各有自己的授權——
`examples/assets/herta/herta.glb` 以 CC-BY 4.0 散布
（上傳者為 Sketchfab 上的 X9_YT；角色「The Herta」© HoYoverse，
依其 Fan Content Guidelines 使用——完整聲明見
`examples/assets/herta/NOTICE.md`）。MMD 演示
`examples/assets/march7th/march7th.pmx` 單獨以 CC-BY 4.0 散布
（上傳者 Gregman；角色「March 7th」© HoYoverse）——見
`examples/assets/march7th/NOTICE.md`。
