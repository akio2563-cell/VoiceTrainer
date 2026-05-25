# VoiceTrainer — 声質分析システム

ボイストレーニング用の声質分析アプリ。
良い声のサンプルをOne-Class SVMで学習し、新しい録音をスコアリングする。

## 起動方法

```
install.bat   # 初回のみ（Pythonが必要）
run.bat       # 起動
```

## ファイル構成

```
VoiceTrainer/
├── main.py            — メインUI（tkinter、3タブ）
├── audio_engine.py    — マイク選択・録音・WAV保存/読込（sounddevice）
├── feature_engine.py  — 音声特徴量抽出（librosa）
├── model_engine.py    — One-Class SVMモデル（scikit-learn）
├── requirements.txt   — 依存ライブラリ
├── install.bat        — セットアップ（Pythonパス検出 → venv → pip）
└── run.bat            — 起動
```

## アーキテクチャ

### 分析する特徴量（音色・声質ベース、音程ではない）

| 指標 | 抽出方法 |
|------|---------|
| 倍音の豊かさ | HPSS（harmonic/percussive分離）による倍音エネルギー比 |
| 声の安定性 | pyin法でf0を推定し、有声区間の標準偏差逆数 |
| 音の明るさ | スペクトル重心（Spectral Centroid） |
| 声の明瞭さ | 有声区間比率（voiced frame ratio） |
| 響き | 倍音比 + スペクトルコントラストの合成 |
| エネルギーの安定性 | RMSエネルギーの変動係数逆数 |

内部特徴ベクトル（ML用）：MFCC×20（平均+標準偏差）、スペクトル重心/帯域/ロールオフ、ゼロ交差率、倍音比、RMS、クロマ×12、スペクトルコントラスト×7、ピッチ安定性、有声比率。

### MLアプローチ

- **One-Class SVM**（`sklearn.svm.OneClassSVM`, `kernel='rbf'`, `nu=0.05`）
- 「良い声」サンプルのみで学習（正常系のみの異常検知）
- 最低3サンプル、推奨10サンプル以上
- decision_function の出力を 0〜100 スコアに変換

### UIフロー

1. **設定タブ** — sounddevice でマイク一覧取得、デバイス選択
2. **サンプル収集タブ** — 録音/WAVインポート → 学習実行 → モデル保存/読込
3. **分析タブ** — 録音またはWAV読込 → 特徴抽出 → スコア表示 + 波形/スペクトログラム描画

---

## トラブルシューティング記録

### 1. `./install.ps1` が cmd で動かない
**原因**: `.ps1` は PowerShell 専用。コマンドプロンプトでは認識されない。  
**対応**: `install.bat` / `run.bat` を追加。

### 2. PowerShell で実行ポリシーエラー
```
このシステムではスクリプトの実行が無効になっているため...
```
**原因**: Windows のデフォルト設定で `.ps1` の実行がブロックされる。  
**対応**: `.bat` を使う。またはバイパスで実行：
```
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

### 3. `.bat` を PowerShell から実行すると構文エラー
```
'exe"' は、内部コマンドまたは外部コマンド...
```
**原因**: `for %%P in (` の複数行リストが PowerShell 経由の cmd サブプロセスで壊れる。  
**対応**: `for` ループをやめて `if exist` の連続に書き直す。

### 4. `.bat` の日本語が文字化け
```
'九▽縺九ｊ縺ｾ縺帙ｓ縲・' は...
```
**原因**: Write ツールが UTF-8 で保存するが、cmd の標準は Shift-JIS。`chcp 65001` を先頭に置いても `.bat` ファイル自体の読み取りには間に合わない。  
**対応**: `.bat` 内のテキストをすべて ASCII（英語）に統一する。

### 5. pip アップグレードエラー
```
ERROR: To modify pip, please run the following command:
python.exe -m pip install --upgrade pip
```
**原因**: Windows の venv 内では `pip install --upgrade pip` は自己更新できない。  
**対応**: `.bat` 内を `python -m pip install --upgrade pip` に変更。
