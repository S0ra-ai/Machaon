# 関連研究

このアイデアは 5 つのコミュニティの交差点にあります。どれもループ全体を出荷しているわけではなく、Machaon は編集・トランザクション・受け入れを 1 つの操作可能な全体に接続します。

## パラメータレベルの知識編集

| 研究 | メモ |
|---|---|
| [ROME](https://arxiv.org/abs/2202.05262) | rank-one 事実編集。この路線の起点 |
| [MEMIT](https://arxiv.org/abs/2210.07229) | 数千の事実をバッチ編集 |
| [MEND](https://arxiv.org/abs/2110.11309) | 編集を予測するハイパーネットワーク |
| [SERAC](https://arxiv.org/abs/2206.06520) | 重みではなく外部メモリ |
| [Survey](https://arxiv.org/abs/2310.19704) | 文献への最良の入口 |
| [EasyEdit](https://github.com/zjunlp/EasyEdit) | 最も近い既存フレームワーク（2.9k★）。バッチスクリプト志向 |

## 訓練不要の重み手術

| 研究 | メモ |
|---|---|
| [Model Surgery](https://arxiv.org/abs/2407.08770) | ごく少数のパラメータで振る舞いが変わる |
| [SCALPEL](https://arxiv.org/abs/2601.07411) | 能力アブレーションのための低ランクパラメータ編集 |
| [ThinkEdit](https://arxiv.org/abs/2503.22048) | 重み編集が推論長を変える |
| [Resisting Harmful Instructions through Training-Free Weight Editing](https://arxiv.org/abs/2610.21189) | 実用的であること、そして危険であることの証拠 |
| [TARS](https://arxiv.org/abs/2412.10257) | 知識除去のための標的角度反転 |

## 表現・活性化編集

| 研究 | メモ |
|---|---|
| [Refusal is mediated by a single direction](https://arxiv.org/abs/2406.11717) | 「1 方向、1 振る舞い」の結果 |
| [RepE](https://arxiv.org/abs/2310.01405) | 表現工学 |
| [CAA](https://arxiv.org/abs/2312.06681) | 対比活性化加算 |
| [ActAdd](https://arxiv.org/abs/2306.03341) | 活性化工学 |
| [SAELens](https://github.com/decoderesearch/SAELens) | スパースオートエンコーダ特徴を編集対象に |

## 重み空間代数

| 研究 | メモ |
|---|---|
| [Task Arithmetic](https://arxiv.org/abs/2212.04089) | タスクベクトル加算でモデルを編集 |
| TIES / DARE | 干渉を意識したマージ |
| [mergekit](https://github.com/arcee-ai/mergekit) | 成熟したツールチェーン（7.4k★） |
| [Evolutionary model merge](https://github.com/SakanaAI/evolutionary-model-merge) | マージレシピの探索 |

## LLM 駆動最適化

| 研究 | メモ |
|---|---|
| [OPRO](https://arxiv.org/abs/2309.03409) | LLM を最適化器として使う |
| [TextGrad](https://arxiv.org/abs/2406.07496) | テキスト勾配 |
| [SEAL](https://arxiv.org/abs/2506.10943) | モデルが自身の更新指示を生成 |
| [Foundation models understand weights](https://arxiv.org/abs/2503.00838) | 「LLM は重み空間を推論できる」を支持 |
| [LLM as a Tool, Not an Agent](https://arxiv.org/abs/2604.16555) | agentic editing への最も鋭い反論 |
| [Agentic NAS](https://arxiv.org/abs/2607.07984) | 対立陣営 |

## 作る前に読むべき 2 本

1. [Revisiting Parameter-Based Knowledge Editing: Theoretical Limits and Empirical Evidence](https://arxiv.org/abs/2606.00570) —— パラメータ編集には根本的な上限があると論じる。
2. [Long-form evaluation of model editing](https://arxiv.org/abs/2402.09394) —— 短い QA を通る編集が長文生成で崩れる。

どちらも `machaon.hygieia` のゲート設計の耐力壁です。

## 空白

1. agent が操作できる編集行動空間がなく、既存は Python API かスクリプト。
2. トランザクション層がなく、スナップショット・検証・ロールバックを担うものがない。
3. 評価が一等市民ではない。
4. LLM が**手術を計画**する系がない：診断 → 部位選択 → 制約宣言 → 閉形式ソルバーへ引き渡し。
