# AI 對話紀錄 — DRL HW2 Cliff Walking

**日期:** 2026-04-21
**Model:** Claude Opus 4.7 (1M context) via Claude Code
**專案:** https://github.com/oomao/HW2_Cliff_Walking
**Live demo:** https://oomao.github.io/HW2_Cliff_Walking/

---

## 使用者初始需求

> 你好 這是一個來自DRL課程的作業2
> 我想要根據 requirement 來完成這份作業 附圖兩個是參考圖 並且希望會使用到 openspec
>
> 交付文件如下
> 1. 該專案成果 (上傳至 github https://github.com/oomao/HW2_Cliff_Walking.git 且使用者為 csm088220@gmail.com 不能有 claude 共同 commit)
> 2. 與 ai 的對話紀錄 (可儲存於本地 ai_record.md 再經由我上傳)
>
> 補充：github 要有 live demo 會更好

參考資料：
- `requirement.txt` — Cliff Walking 作業規格（中文）
- `cliff.jpg` — Q-learning 與 SARSA 的最終策略（參考圖）
- `result_sample.jpg` — SARSA vs Q-learning 獎勵曲線（參考圖）
- Lecture 9B Openspec 優化投影片（openspec 工作流程）

---

## 關鍵決策（詢問使用者後）

| 問題 | 使用者選擇 |
|---|---|
| 實作語言 | Python（numpy + matplotlib） |
| 是否直接 git push | 使用 oomao 身份 push，無 Claude 共同作者；失敗則手動 |
| 是否嚴格依照 openspec 工作流程 | 是 |
| Live demo | 補充要求：GitHub Pages 互動頁面 + 動態 GIF |

---

## 執行步驟

### 1. openspec 工作流程
- 執行 `openspec init`，產生 `.claude/`、`openspec/changes/`、`openspec/specs/`
- 建立 change `01-implement-cliff-walking`（依圖片規範以 `01-` 編號）：
  - `proposal.md` — 為何做、做什麼、影響範圍
  - `tasks.md` — 8 個工作段落的任務清單
  - `design.md` — 環境/演算法/訓練協定的設計筆記
  - `specs/cliff-walking/spec.md` — 6 個需求（環境、Q-learning、SARSA、訓練協定、視覺化、Live demo）
- `openspec validate --strict` 通過

### 2. 實作（Python）
檔案結構：
```
src/cliff_walking/
├── __init__.py
├── env.py      # 4x12 gridworld, Sutton & Barto Example 6.6 語意
├── agents.py   # QLearningAgent / SarsaAgent，共用 ε-greedy 基底
├── train.py    # 50 seeds × 500 episodes 訓練
└── plots.py    # 獎勵曲線、策略 arrow grid、動畫 GIF
```

關鍵設計：
- 狀態：`row * 12 + col`，共 48 個狀態
- 動作：`0=up, 1=right, 2=down, 3=left`
- 獎勵：每步 -1、掉懸崖 -100 並回到起點（不終止）、到終點 0 並終止
- ε-greedy：平手隨機挑選（避免偏向 action 0）
- 超參數：α=0.5, γ=1.0, ε=0.1（參考 Sutton 標準以重現教科書圖形；另提供 CLI flag 支援需求書的 α=0.1, γ=0.9）
- 步數上限：每回合 500 步（避免早期病態無限迴圈）

### 3. 訓練結果（最後 50 回合平均）
- **SARSA:** ≈ -25
- **Q-learning:** ≈ -52

與參考圖 (`result_sample.jpg`) 的形狀一致：
- SARSA 在頂部繞遠路（安全），最終獎勵較高
- Q-learning 貼著懸崖邊緣（最佳理論路徑），但 ε-greedy 造成的偶發掉落拉低平均獎勵

### 4. 視覺化產出 (`artifacts/`)
- `reward_curve.png` — 兩演算法的平滑獎勵曲線
- `policy_qlearning.png` / `policy_sarsa.png` — 最終貪婪策略箭頭網格
- `rollout_qlearning.gif` / `rollout_sarsa.gif` — 從 Start 到 Goal 的動畫
- `q_Q.npy` / `sarsa_Q.npy` — 最終 Q-table
- `q_rewards.npy` / `sarsa_rewards.npy` — 原始獎勵軌跡（50×500）

### 5. Live demo (`docs/`)
- `index.html` — 深色主題，嵌入所有圖表與 GIF
- `assets/` — 圖片複本
- 可透過 GitHub Pages 設定 `main` branch 的 `/docs` 資料夾作為來源

### 6. Dev 腳本 (`scripts/`)
依投影片規範建立：
- `startup.sh` — `git pull`、顯示 handover、`openspec list`、建議下一步
- `ending.sh` — 驗證所有 change、寫入 handover、可選自動 push

### 7. openspec archive
執行 `openspec archive 01-implement-cliff-walking --yes`，產出：
- 移動到 `openspec/changes/archive/2026-04-21-01-implement-cliff-walking/`
- 建立 `openspec/specs/cliff-walking/spec.md`（合併後的累積規格）

### 8. Git commit & push
- `git init -b main`
- 設定 `user.name=oomao`, `user.email=csm088220@gmail.com`
- **無 `Co-Authored-By: Claude` trailer**（依使用者要求）
- 推送到 `https://github.com/oomao/HW2_Cliff_Walking.git`（`main` branch）

---

## 最終分析（寫在 README）

**收斂速度:** 兩者都在前 ~80 回合學完主要策略。SARSA 起始稍低（因其目標依行為動作，探索密集時更噪雜），但之後穩定在更高水準。

**策略差異:**
- Q-learning：學到理論上的最優策略（沿懸崖邊走，13 步），但 ε-greedy 隨機動作有約 25% 機率踩懸崖
- SARSA：把 ε 探索的風險內化到價值估計中，選擇繞遠路（15 步）的安全路徑

**穩定性:** Q-learning 的每回合獎勵波動明顯較大（一次掉懸崖就抹平許多正常步數的獎勵）；SARSA 波動較小。

**選擇建議:**
- 若部署時關閉探索（ε=0），只在乎漸近最優路徑 → Q-learning
- 若部署時仍有探索或雜訊，資料分布與行為一致 → SARSA

---

## 交付檢核

- [x] openspec 工作流程（以 `01-` 編號建立、驗證、歸檔）
- [x] Q-learning 與 SARSA 於 Cliff Walking 的實作
- [x] 50 seeds × 500 episodes 訓練
- [x] 獎勵曲線（符合參考圖形狀）
- [x] 策略箭頭網格（Q-learning 貼懸崖、SARSA 繞頂部，符合參考圖）
- [x] 動畫 GIF（live demo）
- [x] GitHub Pages 網站 (`docs/index.html`)
- [x] README 含理論、參數、結果、分析
- [x] `scripts/startup.sh` + `scripts/ending.sh`
- [x] Commit 作者為 `oomao <csm088220@gmail.com>`，無 Claude trailer
- [x] 推送到 `https://github.com/oomao/HW2_Cliff_Walking.git`

---

## 剩餘手動動作（使用者）

1. 到 GitHub repo → Settings → Pages → Source 選 `main` branch `/docs` 資料夾，啟用後 live demo 會在 `https://oomao.github.io/HW2_Cliff_Walking/` 上線。
