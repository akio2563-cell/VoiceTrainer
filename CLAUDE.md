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
├── audio_engine.py    — マイク選択・録音・WAV保存/読込
├── feature_engine.py  — 音声特徴量抽出
├── model_engine.py    — One-Class SVMモデル
├── requirements.txt   — 依存ライブラリ
├── install.bat        — セットアップ
└── run.bat            — 起動
```

---

## 使用ライブラリ 詳細

### ライブラリ一覧（提供元・ベース・ライセンス）

| ライブラリ | 提供元 | ベース・由来 | ライセンス | GitHub |
|---|---|---|---|---|
| sounddevice | Matthias Geier（個人） | PortAudio（C言語のクロスプラットフォーム音声I/Oライブラリ）のPythonバインディング | MIT | [spatialaudio/python-sounddevice](https://github.com/spatialaudio/python-sounddevice) |
| librosa | Brian McFee ほか（音楽情報検索コミュニティ） | numpy/scipy/matplotlib をベースに音楽・音声分析用APIを構築。もとはニューヨーク大学の研究プロジェクト | ISC | [librosa/librosa](https://github.com/librosa/librosa) |
| scikit-learn | INRIA（フランス国立情報学研究所）発、現在はコミュニティ主導 | SciPy/numpy をベースにした汎用機械学習ライブラリ。2007年 David Cournapeau が Google Summer of Code で開始 | BSD-3-Clause | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) |
| numpy | NumPy Steering Council（コミュニティ） | もとは Numeric（1995年）→ numarray → NumPy（2006年）。多次元配列演算のデファクトスタンダード | BSD | [numpy/numpy](https://github.com/numpy/numpy) |
| matplotlib | John D. Hunter（2003年, MATLAB のグラフAPIに触発されて開発） | MATLAB のplot APIを模倣してPythonで実装。現在はコミュニティ主導 | PSF互換（独自） | [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) |
| scipy | SciPy community（Travis Oliphant ほか, 2001年〜） | numpy を科学計算用に拡張。信号処理・統計・最適化などを含む | BSD | [scipy/scipy](https://github.com/scipy/scipy) |
| joblib | scikit-learn チーム | もとは joblib として独立開発、現在は scikit-learn プロジェクト傘下。piplineの並列化とキャッシュが目的 | BSD-3-Clause | [joblib/joblib](https://github.com/joblib/joblib) |

#### PortAudio（sounddeviceのベース）について
sounddevice は内部で **PortAudio** というC言語ライブラリを呼び出している。
PortAudio はWindows（WASAPI/DirectSound/MME）・Mac（CoreAudio）・Linux（ALSA/JACK）の
各OSの音声ドライバを統一したAPIで扱えるようにした中間層。
1999年 Ross Bencina と Phil Burk が開発、MIT ライセンスのオープンソース。

#### One-Class SVM のアルゴリズム出典
Schölkopf, B. et al. (2001). "Estimating the support of a high-dimensional distribution."
Neural Computation, 13(7), 1443–1471.
異常検知・新規性検出の分野で広く使われる手法。

---

### sounddevice
**役割**: マイクからの録音・デバイス管理

Pythonから直接OSの音声ドライバ（Windows: WASAPI/DirectSound、Mac: CoreAudio）に
アクセスするライブラリ。

```python
# デバイス一覧取得
devices = sd.query_devices()  # マイク名・チャンネル数・サンプリングレートを返す

# ストリーム録音（コールバック方式）
def callback(indata, frames, time, status):
    # 録音中、一定フレームごとに呼ばれる
    chunks.append(indata.copy())

stream = sd.InputStream(device=device_id, channels=1,
                        samplerate=22050, callback=callback)
stream.start()
```

コールバック方式のため、録音中もUIがフリーズしない（別スレッドで動作）。
出力はfloat32の numpy 配列（-1.0〜1.0）。

---

### librosa
**役割**: 音声の特徴量抽出エンジン。このアプリの分析処理の中心。

音楽情報検索（MIR）用のPythonライブラリ。内部でFFT（高速フーリエ変換）を使い、
時間領域の音声波形を周波数領域に変換して様々な特徴を取り出す。

#### MFCC（Mel-Frequency Cepstral Coefficients）
```python
mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=20)
# shape: (20, 時間フレーム数)
```
人間の聴覚特性（メル尺度）に基づいた音の「質感・テクスチャ」を表す係数。
20次元 × 各フレームの平均・標準偏差 = 40次元の特徴量としてMLに渡す。
「この声は柔らかいか・硬いか・鼻にかかっているか」などを数値化する。

#### HPSS（Harmonic-Percussive Source Separation）
```python
harmonic, percussive = librosa.effects.hpss(audio)
harmonic_ratio = mean(|harmonic|) / mean(|audio|)
```
音声を「倍音成分（harmonic）」と「ノイズ・打音成分（percussive）」に分離する。
倍音比が高い = 芯のある、豊かな声。低い = 息漏れ・雑音が多い声。

#### スペクトル重心（Spectral Centroid）
```python
sc = librosa.feature.spectral_centroid(y=audio, sr=22050)
# 単位: Hz
```
全周波数成分の「重心」＝エネルギーが集中している周波数帯。
高い（3000Hz以上）= 明るく通る声。低い（1000Hz以下）= 暗く篭った声。

#### スペクトルコントラスト（Spectral Contrast）
```python
contrast = librosa.feature.spectral_contrast(y=audio, sr=22050)
# shape: (7, 時間フレーム数)  7つの周波数帯ごとの値
```
各周波数帯における「山（倍音のピーク）と谷（倍音間）の差」。
差が大きい = 倍音がはっきり立っている = 響きのある声。

#### pyin（Probabilistic YIN）
```python
f0, voiced_flag, voiced_probs = librosa.pyin(
    audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
)
```
フレームごとの基本周波数（f0 = 声の高さ）を推定するアルゴリズム。
voiced_flag で有声区間（声が出ている部分）と無声区間を分離する。
有声区間のf0の標準偏差 → ピッチの揺れ → 声の安定性の計算に使う。

#### RMS Energy
```python
rms = librosa.feature.rms(y=audio)
```
フレームごとの音量（Root Mean Square）。
標準偏差/平均 = 変動係数 → エネルギーの安定性の計算に使う。

---

### scikit-learn
**役割**: 機械学習モデルの学習・推論

#### StandardScaler（前処理）
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 平均0・標準偏差1に正規化
```
特徴ベクトルの各次元はスケールがバラバラ（MFCCは-200〜200、harmonic_ratioは0〜1など）。
SVMはスケールに敏感なため、正規化が必須。

#### One-Class SVM
```python
from sklearn.svm import OneClassSVM
model = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
model.fit(X_scaled)          # 良い声サンプルのみで学習
score = model.decision_function(X_new)  # 正 → 良い声, 負 → 悪い声
```
通常のSVMは「A vs B」の2クラス分類だが、One-Class SVMは「正常 vs 異常」の1クラス分類。
良い声サンプルが作る特徴空間の「境界球」を学習し、
新しい録音がその球の内側（良い声）か外側（良くない声）かを判定する。

`nu=0.05` = 訓練データの5%を外れ値として許容する（過学習防止）。
`kernel='rbf'` = RBFカーネル（非線形の複雑な境界を表現できる）。

#### 学習フロー
```
良い声 WAV × N本
  ↓ librosa で特徴抽出（各サンプル → 約100次元のベクトル）
  ↓ StandardScaler で正規化
  ↓ OneClassSVM.fit()
  → 学習済みモデル（.pkl として保存可能）

新しい録音
  ↓ 同じ特徴抽出 → 同じ正規化
  ↓ decision_function()
  → スコア（0〜100）
```

---

### numpy
**役割**: 数値計算の基盤

音声データはすべてnumpy配列（ndarray）として扱う。
librosa・scikit-learnも内部的にnumpyを使用しており、データの受け渡しに使う。

```python
# 例: 録音チャンクを結合
audio = np.concatenate(chunks, axis=0).flatten()

# 例: WAV保存用にfloat32→int16変換
int16 = np.clip(audio, -1.0, 1.0)
int16 = (int16 * 32767).astype(np.int16)
```

---

### matplotlib
**役割**: 波形・スペクトログラムの描画

tkinterのウィンドウ内にグラフを埋め込む（`FigureCanvasTkAgg`）。

```python
# 波形
ax.plot(time_axis, audio, linewidth=0.4)

# スペクトログラム（STFT → dB変換 → カラーマップ表示）
D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
librosa.display.specshow(D, sr=22050, x_axis='time', y_axis='hz', cmap='magma')
```

スペクトログラムは「横軸=時間、縦軸=周波数、色=強度」の2Dマップ。
倍音が出ているときは縦に等間隔の明るい帯が並ぶ。

---

### scipy
**役割**: WAVファイルの保存

```python
import scipy.io.wavfile as wavfile
wavfile.write(filepath, samplerate, audio_int16)
```
読み込みはlibrosaを使用（リサンプリング機能があるため）。

---

### joblib
**役割**: 学習済みモデルの保存・読み込み

```python
import joblib
joblib.dump({'scaler': scaler, 'svm': model}, 'model.pkl')  # 保存
data = joblib.load('model.pkl')                              # 読み込み
```
pickle互換だが大きなnumpy配列の保存が高速。

---

## アーキテクチャ

### 分析する特徴量（音色・声質ベース、音程ではない）

| 指標 | 計算式 | 意味 |
|------|--------|------|
| 倍音の豊かさ | `harmonic_ratio × 120` | HPSSで分離した倍音エネルギーの比率 |
| 声の安定性 | `(1 / (f0_std + 1)) × 150` | 有声区間のピッチ標準偏差の逆数 |
| 音の明るさ | `(sc_mean - 500) / 4000 × 100` | スペクトル重心のHz値 |
| 声の明瞭さ | `voiced_ratio × 100` | 有声区間の割合 |
| 響き | `harmonic_ratio × 60 + contrast_mean × 4` | 倍音比＋スペクトルコントラスト |
| エネルギーの安定性 | `100 - (rms_std/rms_mean) × 150` | RMS変動係数の逆数 |

**既知の問題**: 個別バーは固定計算式のため、良いサンプルと比較していない。
総合スコア（One-Class SVM）だけが学習データを反映している。

内部特徴ベクトル（ML用）：MFCC×20（平均+標準偏差）、スペクトル重心/帯域/ロールオフ、
ゼロ交差率、倍音比、RMS、クロマ×12、スペクトルコントラスト×7、ピッチ安定性、有声比率。
合計約100次元。

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

### 6. 声の安定性スコアが常に低い
**原因**: スコア計算の係数が `× 12` で小さすぎた。安定した声でも最高8点程度にしかならなかった。  
**対応**: `× 150` に修正。f0標準偏差 1Hz → 75点、2Hz → 50点 の範囲に正規化。
