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
  gaits、body trajectories、morph 時間線，以及行內 expression DSL。
  詳見 [`docs/declarative_animation.md`](docs/declarative_animation.md)。
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

## 專案結構

```
PoseCascade/
├── posecascade/                  # 主套件
│   ├── animation/                # 布料、蒙皮、morph、IK、VMD 軌道
│   │   ├── cloth.py              # PBD 求解器（Python 編排層）
│   │   └── _cloth_kernels.pyx    # Cython 內迴圈（由 setup.py 建構）
│   ├── app/                      # QApplication bootstrap、主視窗
│   ├── assets/                   # 快取、路徑安全、importer 管理員
│   ├── gl/                       # GL context、shader、framebuffer
│   ├── mcp/                      # Model Context Protocol 伺服器
│   ├── render/                   # 渲染圖、材質、燈光
│   ├── scene/                    # scene graph、transform、component
│   ├── scripting/                # 沙箱腳本主機 + 聲明式 runtime
│   └── ui/                       # viewport、outliner、inspector、timeline
├── importers/<format>/           # 每個格式各自的 importer 外掛
├── shaders/                      # 按 render pass 分類的 GLSL
├── examples/                     # 隨附模型 + 動畫腳本
├── tests/                        # pytest 測試套件，鏡射套件結構
├── docs/                         # 設計 + 整合文件
├── schemas/                      # JSON schema（聲明式動畫）
├── setup.py                      # cythonize build hook
└── pyproject.toml                # 專案 metadata + ruff / bandit 設定
```

## 開發

[`CLAUDE.md`](CLAUDE.md) 裡的 Definition of Done 要求每個變更在 commit 之前
都要通過三道閘門。本地重現：

```bash
.venv/Scripts/python.exe -m pytest tests/             # unit + offscreen-GL 測試
.venv/Scripts/python.exe -m ruff check .              # lint + style
.venv/Scripts/python.exe -m bandit -c pyproject.toml -r posecascade/
```

布料 Cython kernel 每次改 `.pyx` 原始碼後都要重新就地建構：

```bash
.venv/Scripts/python.exe setup.py build_ext --inplace
```

要散布的時候，`pyproject.toml` 裡的 `[tool.cibuildwheel]` 區段會在
Win / macOS / Linux × 各支援 Python 版本上產出預建 wheel；
`.github/workflows/wheels.yml` 在 tag push 時觸發那個流程。

## 視覺管線

前向渲染器每幀依固定順序跑六個 pass —— depth-map 陰影 pass、scene、地面、
projected shadow、selection overlay、後處理 effect chain。每個 pass 都有開關
（`set_ground_enabled`、`set_self_shadow_enabled`、`set_projected_shadow_enabled`、
`set_selected_holder`），這樣 smoke test 與 headless render 可以選擇關掉某些 pass
而不影響其他 pass 的像素保真度。完整拆解——pass 順序、shader 檔案、light-space
數學、texture unit、MMD-fluence gap——都在
[`docs/rendering_pipeline.md`](docs/rendering_pipeline.md)。

## 效能備註

布料求解器是近期主要的優化重點。`posecascade.mcp.server.cloth_benchmark`
裡 480-vert 裙子的 benchmark：

| 階段                                   | ms/step (best) |  vs baseline |
|----------------------------------------|---------------:|-------------:|
| Baseline（優化前）                     |          3.225 |            — |
| NumPy：einsum + 合併 bincount          |          2.085 |         −35% |
| Cython kernel                          |          0.356 |     **−89%** |
| Cython + broad-phase + bin culling     |  0.36–0.38（單 bin collider 多省 30%） | — |

渲染器熱路徑的每幀計時都用 `posecascade.utils.profiling.frame_section` 包起來，
UI overlay（或自訂測試）能從 `current_stats().sections` 拉出每幀分解。

## 授權

專案的 MIT-style 條款見 [`LICENSE`](LICENSE)。隨附素材各有自己的授權——
`examples/assets/herta/herta.glb` 以 CC-BY 4.0 散布
（上傳者為 Sketchfab 上的 X9_YT；角色「The Herta」© HoYoverse，
依其 Fan Content Guidelines 使用——完整聲明見
`examples/assets/herta/NOTICE.md`）。MMD 演示
`examples/assets/march7th/march7th.pmx` 單獨以 CC-BY 4.0 散布
（上傳者 Gregman；角色「March 7th」© HoYoverse）——見
`examples/assets/march7th/NOTICE.md`。
