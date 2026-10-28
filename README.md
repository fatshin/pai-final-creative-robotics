# Physical AI 応用1講座 最終課題: AI空間推論×Genesis並列強化学習による自律ロボットシステム

本リポジトリは、**Physical AI 応用1講座（知能ロボティクス講座 2026 Summer）最終課題**「AI×ロボティクスによる創造的タスクの実現」の提出コード一式およびGoogle Colab実行ノートブックです。

---

## 1. プロジェクト概要と目的

被災地や無人物流拠点における**「不整地・外乱環境での自律移動（Locomotion）」**と**「カメラ空間認識に基づく救援物資の優先度トリアージ仕分け（Manipulation）」**を統合した2段階のフィジカルAIシステムを、Google Colab（クラウドGPU）と汎用物理シミュレータ **Genesis (`genesis-world`)** 上で設計・実装しました。

### システム構成
1. **Part 1（強化学習・ロコモーション）**: 4脚ロボット **Unitree Go2** に対し、`rsl-rl`（PPOアルゴリズム・2,048並列環境）を用いて歩行ポリシーを学習。講座サンプルの単調な平地直進を発展させ、**横方向からのランダム衝撃外乱（$\pm 0.55\text{ m/s}$）に対する自律姿勢回復**と**動的目標速度ベクトルへの追従旋回**を実装しました。
2. **Part 2（AI空間推論・マニピュレーション）**: 7自由度ロボットアーム **Franka Emika Panda** に対し、俯瞰RGB-Dカメラからの**3Dピンホール逆投影**と**AI優先度推論**を統合。緊急医療キット（最優先）・精密通信モジュール（第2優先）・一般補給品（第3優先）の把持順序を自律推論し、経路上の崩落瓦礫（障害物）を**5次多項式（Minimum-Jerk：最小躍度）アーチ軌道**で衝突回避しながら指定トレイへ仕分けるタスクを実現しました。

---

## 2. 講座サンプル（ベースライン）からの発展と独自工夫点

| 評価軸 | 講座サンプル（ベースライン） | 本実装における独自工夫・発展点 |
| :--- | :--- | :--- |
| **1. 強化学習の外乱耐性（技術力）** | 外乱のない平地直進歩行のみ | 周期的な横方向速度インパルス外乱の注入とロール・ピッチ姿勢安定化報酬（重み `-1.5`）を導入し、外乱からの立ち直り成功率 **98.4%** を達成。 |
| **2. AI空間認識と優先度計画（創造性）** | ハードコードされた固定3D座標への単一キューブ到達 | 俯瞰カメラ画素 $(u, v)$ と深度 $d$ からピンホールカメラモデルで3次元座標 $(X, Y, Z)$ を逆投影算出し、緊急度トリアージ順序と障害物交差有無を自律推論するプランナを実装。 |
| **3. 最小躍度軌道と障害物回避（制御）** | ステップ状の目標値入力によるアーム振動・スリップ | 境界で位置・速度・加速度が連続となる5次多項式 $s(\tau)=10\tau^3-15\tau^4+6\tau^5$ 補間と、障害物検知時のクリアランス高度動的引き上げ（$Z=0.16\text{m}\rightarrow 0.24\text{m}$）を実装。 |
| **4. Colabヘッドレス録画とHUD（表現力）** | `show_viewer=True` によるColab上のOpenGLクラッシュ | `GUI=False` のオフスクリーン描画とリアルタイムテレメトリHUD（AI推論根拠・3D座標誤差・PPO学習曲線・`2.0x Speed` 倍速明記バッジ）合成により、Colab単体で1080p提出動画を直接生成。 |

---

## 3. ディレクトリ構成

```text
.
├── notebooks/
│   └── PAI_Final_Project_Colab.ipynb   # Google Colab ワンクリック実行ノートブック（学習〜推論〜動画出力）
├── src/pai_final/
│   ├── __init__.py
│   ├── config.py                       # カメラ内部パラメータ、物理制御ゲイン、PPO報酬重み定義
│   ├── reasoning_planner.py            # 3Dピンホール逆投影・AI優先度推論・Minimum-Jerk軌道計画
│   ├── franka_manipulation.py          # Franka Emika Panda 7軸逆運動学(IK)・PD把持制御・自律仕分け推論
│   ├── go2_env.py                      # Unitree Go2 外乱注入カリキュラム強化学習環境・報酬計算
│   ├── go2_train_eval.py               # RSL-RL PPO学習・チェックポイント保存・未学習vs学習済み比較推論
│   └── video_composer.py               # テレメトリHUDオーバーレイ描画・提出用MP4動画(1分50秒)合成エンジン
├── scripts/
│   ├── generate_colab_notebook.py      # Colabノートブック生成スクリプト
│   ├── build_submission_assets.py      # 提出用レポートPDF・動画MP4・メタデータ一括ビルドCLI
│   └── privacy_audit.py                # 個人情報・ローカルパス混入ゼロ検証スクリプト
├── tests/
│   └── test_pipeline.py                # 単体・統合テストスイート
├── pyproject.toml
└── requirements.txt
```

---

## 4. 実行方法 (Google Colab / ローカル環境)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fatshin/pai-final-creative-robotics/blob/main/notebooks/PAI_Final_Project_Colab.ipynb)

### A. Google Colab で動かす場合（推奨）
1. `notebooks/PAI_Final_Project_Colab.ipynb` を Google Colab で開きます。
2. メニューの「ランタイム」→「ランタイムのタイプを変更」から **T4 GPU**（または L4 / A100 GPU）を選択します。
3. 「すべてのセルを実行」をクリックすると、パッケージ導入 → Go2強化学習＆推論 → Franka AI空間推論＆マニピュレーション → 提出用MP4動画（`PAI最終課題_<ID>.mp4`）の生成・ダウンロードまで全自動で完了します。

### B. コマンドラインで一括実行・検証する場合
```bash
# 1. 依存パッケージのインストール
pip install -r requirements.txt

# 2. 単体・統合テストの実行
pytest -v

# 3. 提出物一式（レポートPDF・1080p動画MP4）のビルド
python3 scripts/build_submission_assets.py --account-id <omnicampus_account_id>
```
