# HW2 — Cliff Walking：Q-learning 與 SARSA 比較

DRL 作業 2。在經典的 4×12 Cliff Walking 格子世界（Sutton & Barto 第二版 Example 6.6）
上，實作並比較 Q-learning（離策略）與 SARSA（同策略）兩種表格式強化學習演算法。

**Live demo：** <https://oomao.github.io/HW2_Cliff_Walking/>

---

## 一、環境描述

- 格子大小：4 列 × 12 行，動作結果為確定性，越界時位置會被夾回格子內。
- 起點（Start）：左下角 `(3, 0)`；終點（Goal）：右下角 `(3, 11)`。
- 懸崖（Cliff）：`(3, 1) … (3, 10)`。一旦踩進任一懸崖格，回饋為 **−100**，
  agent 會被傳回 Start，**但回合不會結束**。
- 其他任何一步的回饋皆為 **−1**。
- 動作空間：`{上, 右, 下, 左}`。

## 二、演算法

兩個 agent 共用一個 `Q ∈ ℝ⁴⁸ˣ⁴` 的表格，以及 ε-greedy 的行為策略
（argmax 平手時以均勻機率隨機挑一個，避免偏向某一個動作）。

| | Q-learning（離策略） | SARSA（同策略） |
|---|---|---|
| 更新公式 | `Q[s,a] += α (r + γ · max_{a'} Q[s', a'] − Q[s, a])` | `Q[s,a] += α (r + γ · Q[s', a'] − Q[s, a])` |
| 目標中的 `a'` | 下一狀態中「理論上最好」的動作（不一定會真的被執行） | 下一步依 ε-greedy **實際採取**的動作 |

## 三、超參數

| 參數 | 數值 |
|---|---|
| α（學習率） | 0.5 |
| γ（折扣因子） | 1.0 |
| ε（探索率） | 0.1 |
| 每次訓練的 episodes | 500 |
| 獨立種子數（seeds） | 50 |
| 每回合步數上限 | 500 |

作業規格書另外提到 `α=0.1, γ=0.9`，本專案已把這兩個值暴露成 CLI flag
（`--alpha`、`--gamma`）。不過主要圖形採用 Sutton & Barto 標準設定，
如此曲線形狀才能與參考圖 `result_sample.jpg` 對齊。

## 四、結果

### 4.1 獎勵曲線（Total Reward vs Episodes）

![Reward curve](artifacts/reward_curve.png)

每條曲線皆為 **50 個獨立 seed 的平均**，並套用長度 10 的移動平均。
SARSA 因為把 ε-探索帶來的風險反映在自己的價值估計中，最終的回合總獎勵比 Q-learning 高（更接近 0）。

### 4.2 學到的貪婪策略

| Q-learning | SARSA |
|---|---|
| ![Q-learning policy](artifacts/policy_qlearning.png) | ![SARSA policy](artifacts/policy_sarsa.png) |

Q-learning 沿著懸崖正上方那一列走（最短路徑，共 13 步）。
SARSA 則繞到網格頂端（共 15 步），避免一旦 ε 探索觸發隨機動作時直接掉落懸崖。
這正是參考圖 `cliff.jpg` 想呈現的差異。

### 4.3 動畫展示（live rollout）

| Q-learning | SARSA |
|---|---|
| ![Q-learning rollout](artifacts/rollout_qlearning.gif) | ![SARSA rollout](artifacts/rollout_sarsa.gif) |

## 五、理論比較與分析

**收斂速度。** 兩種方法都在前約 80 個回合學完主要策略。SARSA 在初期的移動平均稍低——
因為它的更新目標依賴「實際採取的下一個動作」，當探索密集時估計較為嘈雜——
但之後穩定在更高水準。

**最終表現（在 ε-greedy 行為策略下，取最後 50 個回合平均）：**

| 方法 | 最終平均獎勵 |
|---|---|
| SARSA | **≈ −25** |
| Q-learning | ≈ −52 |

乍看之下違反直覺，但 Q-learning 在訓練過程反而「比較差」。
它學到的是**理論上的最優策略**（貼著懸崖走），可是 ε-greedy 有 10% 機率強制隨機動作，
而貼著懸崖邊的隨機動作有約四分之一機率直接掉下去，吃到 −100 的罰分。
SARSA 因為把這個風險納入價值估計，所以寧可繞遠路。

**穩定性。** Q-learning 的每回合獎勵波動明顯較大，因為一次掉懸崖就會抹平大量正常步的獎勵；
SARSA 的回合獎勵分布較緊湊，整體波動較小。

**探索對結果的影響。** 這就是教科書上 on-policy 與 off-policy 差異最經典的例子：
Q-learning 更新時會使用「如果之後永遠最優地走」的假設，因此它只在乎漸近最優策略是什麼；
SARSA 更新時完全根據「我實際會怎麼走」，因此它會把目前採用的探索行為影響內化進來。

## 六、結論要求對照

- **哪一種收斂較快？** 兩者在本環境的收斂速度相近，前 80 回合就大致穩定；
  Q-learning 的最終策略更貼近理論最優，但 SARSA 的實際回合獎勵更早、更穩定地收斂到較高值。
- **哪一種較穩定？** SARSA；因為它避開高風險區，波動明顯較小。
- **什麼情境下選擇 Q-learning？** 部署時會關閉探索（ε=0），且環境對單次失誤容忍度高、
  只在乎漸近最優路徑的情境。
- **什麼情境下選擇 SARSA？** 部署時仍會保留探索或雜訊、或環境中單次錯誤的成本極高
  （例如實體機器人、安全攸關的控制系統）時更適合使用。

## 七、專案結構

```
.
├── src/cliff_walking/   # 環境、agent、訓練、繪圖
├── artifacts/           # 訓練後產出的圖檔、GIF 與原始 .npy 軌跡
├── docs/                # GitHub Pages live demo（以 /docs 對外服務）
├── scripts/             # startup.sh / ending.sh 開發流程腳本
├── openspec/            # openspec 變更與規格歷史
├── requirements.txt
└── README.md
```

## 八、重現結果

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m cliff_walking.train   # 產生 artifacts/*.npy
PYTHONPATH=src python -m cliff_walking.plots   # 產生 artifacts/*.png, *.gif
```

50 seeds × 500 episodes × 2 演算法，在現代 CPU 上不到一分鐘即可跑完。

## 九、參考資料

- Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*（第二版），
  Example 6.6 — "Cliff Walking"。
