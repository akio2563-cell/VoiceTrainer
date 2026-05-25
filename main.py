"""
ボイストレーニング声質分析システム
良い声のサンプルを学習し、声質をスコアリングするアプリ。
"""

import os
import shutil
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from audio_engine import AudioEngine
from feature_engine import FeatureEngine
from model_engine import VoiceModel

# Japanese font on Windows
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

SAMPLES_DIR = "good_samples"
SCORE_NAMES = ['倍音の豊かさ', '声の安定性', '音の明るさ', '声の明瞭さ', '響き', 'エネルギーの安定性']
BAR_WIDTH = 190  # px


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _score_color(val: float) -> str:
    if val >= 70:
        return '#27ae60'
    if val >= 45:
        return '#f39c12'
    return '#e74c3c'


# ──────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ボイストレーニング声質分析システム")
        self.root.geometry("1000x720")
        self.root.resizable(True, True)

        self.audio = AudioEngine()
        self.features = FeatureEngine()
        self.model = VoiceModel()

        self.selected_device: int | None = None
        self.devices: list[dict] = []

        os.makedirs(SAMPLES_DIR, exist_ok=True)

        self._build_ui()
        self._refresh_devices()

    # ── UI construction ──────────────────────

    def _build_ui(self):
        header = tk.Frame(self.root, bg='#1a252f', pady=8)
        header.pack(fill='x')
        tk.Label(
            header,
            text="ボイストレーニング 声質分析システム",
            font=('Yu Gothic', 15, 'bold'),
            bg='#1a252f', fg='#ecf0f1',
        ).pack()

        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=8, pady=8)

        self.tab_cfg = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_analyze = ttk.Frame(nb)

        nb.add(self.tab_cfg, text='⚙ 設定')
        nb.add(self.tab_train, text='📚 サンプル収集・学習')
        nb.add(self.tab_analyze, text='🎯 リアルタイム分析')

        self._build_settings(self.tab_cfg)
        self._build_training(self.tab_train)
        self._build_analysis(self.tab_analyze)

    # ── Settings tab ─────────────────────────

    def _build_settings(self, parent):
        f = ttk.LabelFrame(parent, text="入力デバイス（マイク）設定", padding=16)
        f.pack(fill='both', expand=True, padx=20, pady=20)

        tk.Label(f, text="使用するマイクを選択してください:", font=('Yu Gothic', 11)).pack(anchor='w')

        lb_frame = tk.Frame(f)
        lb_frame.pack(fill='x', pady=6)
        sb = tk.Scrollbar(lb_frame, orient='vertical')
        self.device_lb = tk.Listbox(
            lb_frame, height=12, font=('Yu Gothic', 10),
            yscrollcommand=sb.set, selectmode='single',
            activestyle='dotbox',
        )
        sb.config(command=self.device_lb.yview)
        self.device_lb.pack(side='left', fill='x', expand=True)
        sb.pack(side='right', fill='y')
        self.device_lb.bind('<<ListboxSelect>>', self._on_device_select)

        btn_row = tk.Frame(f)
        btn_row.pack(fill='x', pady=4)
        ttk.Button(btn_row, text="一覧を更新", command=self._refresh_devices).pack(side='left', padx=4)
        ttk.Button(btn_row, text="3秒テスト録音", command=self._test_device).pack(side='left', padx=4)

        self.cfg_status = tk.Label(f, text="", font=('Yu Gothic', 10))
        self.cfg_status.pack(anchor='w', pady=6)

    def _refresh_devices(self):
        self.devices = self.audio.get_devices()
        default_id = self.audio.get_default_input_id()

        self.device_lb.delete(0, tk.END)
        for i, d in enumerate(self.devices):
            label = f"[{d['id']}] {d['name']}  (ch:{d['channels']})"
            self.device_lb.insert(tk.END, label)
            if d['id'] == default_id:
                self.device_lb.selection_set(i)
                self.device_lb.see(i)
                self.selected_device = d['id']
                self.cfg_status.config(
                    text=f"デフォルト選択: {d['name']}", fg='#27ae60'
                )

    def _on_device_select(self, _event):
        sel = self.device_lb.curselection()
        if sel:
            d = self.devices[sel[0]]
            self.selected_device = d['id']
            self.cfg_status.config(text=f"選択: {d['name']}", fg='#27ae60')

    def _test_device(self):
        if self.selected_device is None:
            messagebox.showwarning("警告", "デバイスを選択してください")
            return
        self.cfg_status.config(text="録音中… (3秒)", fg='#e74c3c')

        def run():
            try:
                self.audio.start_recording(self.selected_device)
                time.sleep(3)
                data = self.audio.stop_recording()
                msg = f"成功: {len(data)} サンプル ({len(data)/self.audio.sample_rate:.1f}秒)"
                self.root.after(0, lambda: self.cfg_status.config(text=msg, fg='#27ae60'))
            except Exception as e:
                self.root.after(0, lambda: self.cfg_status.config(text=f"エラー: {e}", fg='#e74c3c'))

        threading.Thread(target=run, daemon=True).start()

    # ── Training tab ─────────────────────────

    def _build_training(self, parent):
        top = ttk.LabelFrame(parent, text="良い声のサンプル収集", padding=12)
        top.pack(fill='x', padx=16, pady=(16, 4))

        tk.Label(
            top,
            text="ボイストレーナーが「良い」と判断した声を録音・インポートしてください。\n"
                 "最低3サンプル（推奨10以上）を集めてから学習を実行します。",
            font=('Yu Gothic', 10), justify='left',
        ).pack(anchor='w')

        self.sample_count_lbl = tk.Label(top, text="サンプル数: 0", font=('Yu Gothic', 11, 'bold'), fg='#2980b9')
        self.sample_count_lbl.pack(anchor='w', pady=4)

        btn_row = tk.Frame(top)
        btn_row.pack(fill='x', pady=4)

        self.btn_rec_start = ttk.Button(btn_row, text="● 録音開始", command=self._sample_rec_start)
        self.btn_rec_start.pack(side='left', padx=4)

        self.btn_rec_stop = ttk.Button(btn_row, text="■ 停止・保存", command=self._sample_rec_stop, state='disabled')
        self.btn_rec_stop.pack(side='left', padx=4)

        ttk.Button(btn_row, text="WAVファイルをインポート", command=self._import_wavs).pack(side='left', padx=4)

        self.rec_status = tk.Label(top, text="", font=('Yu Gothic', 10))
        self.rec_status.pack(anchor='w')

        # Sample list
        list_frame = ttk.LabelFrame(parent, text="録音済みサンプル", padding=8)
        list_frame.pack(fill='both', expand=True, padx=16, pady=4)

        lf = tk.Frame(list_frame)
        lf.pack(fill='both', expand=True)
        sb2 = tk.Scrollbar(lf)
        self.sample_lb = tk.Listbox(lf, font=('Yu Gothic', 10), yscrollcommand=sb2.set)
        sb2.config(command=self.sample_lb.yview)
        self.sample_lb.pack(side='left', fill='both', expand=True)
        sb2.pack(side='right', fill='y')

        ttk.Button(list_frame, text="選択を削除", command=self._delete_sample).pack(side='left', padx=4)

        # Training controls
        train_frame = ttk.LabelFrame(parent, text="モデル学習", padding=10)
        train_frame.pack(fill='x', padx=16, pady=(4, 16))

        tf_row = tk.Frame(train_frame)
        tf_row.pack(fill='x')
        self.btn_train = ttk.Button(tf_row, text="学習を実行", command=self._train)
        self.btn_train.pack(side='left', padx=4)
        ttk.Button(tf_row, text="モデルを保存", command=self._save_model).pack(side='left', padx=4)
        ttk.Button(tf_row, text="モデルを読込", command=self._load_model).pack(side='left', padx=4)

        self.train_status = tk.Label(train_frame, text="", font=('Yu Gothic', 10))
        self.train_status.pack(anchor='w', pady=4)

        self._refresh_sample_list()

    def _sample_rec_start(self):
        if self.selected_device is None:
            messagebox.showwarning("警告", "設定タブでマイクを選択してください")
            return
        self.audio.start_recording(self.selected_device)
        self.btn_rec_start.config(state='disabled')
        self.btn_rec_stop.config(state='normal')
        self.rec_status.config(text="● 録音中…", fg='#e74c3c')

    def _sample_rec_stop(self):
        data = self.audio.stop_recording()
        self.btn_rec_start.config(state='normal')
        self.btn_rec_stop.config(state='disabled')

        if len(data) == 0:
            self.rec_status.config(text="データなし", fg='#e74c3c')
            return

        path = self._next_sample_path()
        self.audio.save_wav(data, path)
        self.rec_status.config(text=f"保存: {os.path.basename(path)}", fg='#27ae60')
        self._refresh_sample_list()

    def _import_wavs(self):
        paths = filedialog.askopenfilenames(
            title="良い声のWAVファイルを選択",
            filetypes=[("WAV", "*.wav"), ("全て", "*.*")],
        )
        for p in paths:
            shutil.copy2(p, self._next_sample_path())
        if paths:
            self._refresh_sample_list()
            messagebox.showinfo("完了", f"{len(paths)} ファイルをインポートしました")

    def _next_sample_path(self) -> str:
        existing = self._sample_files()
        idx = len(existing) + 1
        while True:
            p = os.path.join(SAMPLES_DIR, f"sample_{idx:04d}.wav")
            if not os.path.exists(p):
                return p
            idx += 1

    def _sample_files(self) -> list[str]:
        if not os.path.isdir(SAMPLES_DIR):
            return []
        return sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith('.wav'))

    def _refresh_sample_list(self):
        files = self._sample_files()
        self.sample_lb.delete(0, tk.END)
        for f in files:
            self.sample_lb.insert(tk.END, f)
        n = len(files)
        suffix = ' ✓ 十分です' if n >= 10 else ' (推奨: 10以上)' if n >= 3 else ' (最低3必要)'
        self.sample_count_lbl.config(text=f"サンプル数: {n}{suffix}")

    def _delete_sample(self):
        sel = self.sample_lb.curselection()
        if not sel:
            return
        name = self.sample_lb.get(sel[0])
        if messagebox.askyesno("削除確認", f"{name} を削除しますか？"):
            os.remove(os.path.join(SAMPLES_DIR, name))
            self._refresh_sample_list()

    def _train(self):
        files = self._sample_files()
        if len(files) < 3:
            messagebox.showwarning("警告", "最低3サンプル必要です")
            return

        self.btn_train.config(state='disabled')
        self.train_status.config(text="学習中…", fg='#f39c12')

        def run():
            try:
                mdl = VoiceModel()
                for i, fname in enumerate(files):
                    path = os.path.join(SAMPLES_DIR, fname)
                    audio = self.audio.load_wav(path)
                    feats = self.features.extract(audio)
                    vec = self.features.to_vector(feats)
                    if vec is not None:
                        mdl.add_sample(vec)
                    msg = f"特徴抽出中… {i+1}/{len(files)}"
                    self.root.after(0, lambda m=msg: self.train_status.config(text=m, fg='#f39c12'))

                mdl.train()
                self.model = mdl
                self.root.after(0, lambda: self.train_status.config(
                    text=f"✓ 学習完了 (サンプル: {mdl.sample_count}件)  → 分析タブへ",
                    fg='#27ae60',
                ))
            except Exception as e:
                self.root.after(0, lambda: self.train_status.config(text=f"エラー: {e}", fg='#e74c3c'))
            finally:
                self.root.after(0, lambda: self.btn_train.config(state='normal'))

        threading.Thread(target=run, daemon=True).start()

    def _save_model(self):
        if not self.model.is_trained:
            messagebox.showwarning("警告", "先に学習を実行してください")
            return
        path = filedialog.asksaveasfilename(
            title="モデルを保存",
            defaultextension=".pkl",
            filetypes=[("モデル", "*.pkl")],
        )
        if path:
            self.model.save(path)
            messagebox.showinfo("完了", "モデルを保存しました")

    def _load_model(self):
        path = filedialog.askopenfilename(
            title="モデルを読み込み",
            filetypes=[("モデル", "*.pkl")],
        )
        if path:
            try:
                self.model.load(path)
                messagebox.showinfo("完了", "モデルを読み込みました")
            except Exception as e:
                messagebox.showerror("エラー", f"読み込み失敗: {e}")

    # ── Analysis tab ─────────────────────────

    def _build_analysis(self, parent):
        outer = tk.Frame(parent)
        outer.pack(fill='both', expand=True)

        # Left panel
        left = tk.Frame(outer, width=260)
        left.pack(side='left', fill='y', padx=8, pady=8)
        left.pack_propagate(False)

        ctrl = ttk.LabelFrame(left, text="録音・分析", padding=8)
        ctrl.pack(fill='x', pady=(0, 8))

        self.btn_an_start = ttk.Button(ctrl, text="● 録音開始", command=self._analysis_start)
        self.btn_an_start.pack(fill='x', pady=3)

        self.btn_an_stop = ttk.Button(ctrl, text="■ 停止して分析", command=self._analysis_stop, state='disabled')
        self.btn_an_stop.pack(fill='x', pady=3)

        ttk.Button(ctrl, text="WAVファイルを分析", command=self._analyze_file).pack(fill='x', pady=3)

        self.an_status = tk.Label(ctrl, text="モデルを学習してから使用", font=('Yu Gothic', 9), wraplength=220)
        self.an_status.pack(pady=4)

        # Score
        score_f = ttk.LabelFrame(left, text="総合スコア", padding=8)
        score_f.pack(fill='x', pady=(0, 8))

        self.score_var = tk.StringVar(value="--")
        tk.Label(score_f, textvariable=self.score_var, font=('Yu Gothic', 40, 'bold'), fg='#2980b9').pack()

        self.verdict_var = tk.StringVar(value="")
        tk.Label(score_f, textvariable=self.verdict_var, font=('Yu Gothic', 11), wraplength=220).pack()

        # Individual bars
        bar_f = ttk.LabelFrame(left, text="声質の各指標", padding=8)
        bar_f.pack(fill='both', expand=True)

        self._bars: dict[str, tuple] = {}
        for name in SCORE_NAMES:
            tk.Label(bar_f, text=name, font=('Yu Gothic', 9), anchor='w').pack(fill='x')
            row = tk.Frame(bar_f)
            row.pack(fill='x', pady=2)

            bg = tk.Frame(row, bg='#dfe6e9', height=14, width=BAR_WIDTH)
            bg.pack(side='left')
            bg.pack_propagate(False)

            fill = tk.Frame(bg, bg='#3498db', height=14, width=0)
            fill.place(x=0, y=0, relheight=1.0)

            val_lbl = tk.Label(row, text="--", font=('Yu Gothic', 9), width=4, anchor='e')
            val_lbl.pack(side='left', padx=3)

            self._bars[name] = (fill, val_lbl, bg)

        # Right panel — plots
        right = tk.Frame(outer)
        right.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        self._fig = Figure(figsize=(6, 5), dpi=90)
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        self._init_plots()

    def _init_plots(self):
        self._fig.clear()
        ax1 = self._fig.add_subplot(211)
        ax1.set_title('波形', fontsize=9)
        ax1.set_xlabel('時間 (秒)', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=7)

        ax2 = self._fig.add_subplot(212)
        ax2.set_title('スペクトログラム', fontsize=9)
        ax2.set_xlabel('時間 (秒)', fontsize=8)
        ax2.set_ylabel('周波数 (Hz)', fontsize=8)
        ax2.tick_params(labelsize=7)

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()

    def _analysis_start(self):
        if self.selected_device is None:
            messagebox.showwarning("警告", "設定タブでマイクを選択してください")
            return
        if not self.model.is_trained:
            messagebox.showwarning("警告", "サンプル収集タブで学習を先に行ってください")
            return
        self.audio.start_recording(self.selected_device)
        self.btn_an_start.config(state='disabled')
        self.btn_an_stop.config(state='normal')
        self.an_status.config(text="● 録音中…", fg='#e74c3c')

    def _analysis_stop(self):
        data = self.audio.stop_recording()
        self.btn_an_start.config(state='normal')
        self.btn_an_stop.config(state='disabled')

        if len(data) == 0:
            self.an_status.config(text="データなし", fg='#e74c3c')
            return

        self.an_status.config(text="分析中…", fg='#f39c12')
        threading.Thread(target=self._run_analysis, args=(data,), daemon=True).start()

    def _analyze_file(self):
        if not self.model.is_trained:
            messagebox.showwarning("警告", "学習を先に行ってください")
            return
        path = filedialog.askopenfilename(
            title="分析するWAVファイル",
            filetypes=[("WAV", "*.wav")],
        )
        if path:
            self.an_status.config(text="分析中…", fg='#f39c12')

            def run():
                audio = self.audio.load_wav(path)
                self._run_analysis(audio)

            threading.Thread(target=run, daemon=True).start()

    def _run_analysis(self, audio: np.ndarray):
        try:
            feats = self.features.extract(audio)
            if feats is None:
                self.root.after(0, lambda: self.an_status.config(
                    text="音声が短すぎます (0.5秒以上必要)", fg='#e74c3c'
                ))
                return

            vec = self.features.to_vector(feats)
            is_good, score = self.model.predict(vec)
            ind_scores = self.features.readable_scores(feats)

            self.root.after(0, lambda: self._display_result(audio, is_good, score, ind_scores))
        except Exception as e:
            self.root.after(0, lambda: self.an_status.config(text=f"エラー: {e}", fg='#e74c3c'))

    def _display_result(self, audio: np.ndarray, is_good: bool, score: float, ind_scores: dict):
        # Overall score
        s = int(score)
        self.score_var.set(str(s))

        if score >= 70:
            verdict, color = "良い声です！", '#27ae60'
        elif score >= 50:
            verdict, color = "まずまずです", '#f39c12'
        else:
            verdict, color = "改善の余地があります", '#e74c3c'
        self.verdict_var.set(verdict)

        # Individual bars
        for name, (fill, val_lbl, bg) in self._bars.items():
            val = float(ind_scores.get(name, 0))
            w = int(BAR_WIDTH * val / 100)
            fill.place(x=0, y=0, relheight=1.0, width=w)
            fill.config(bg=_score_color(val))
            val_lbl.config(text=str(int(val)))

        # Plots
        self._fig.clear()
        sr = self.audio.sample_rate

        ax1 = self._fig.add_subplot(211)
        t = np.linspace(0, len(audio) / sr, len(audio))
        ax1.plot(t, audio, linewidth=0.4, color='#2980b9')
        ax1.set_title('波形', fontsize=9)
        ax1.set_xlabel('時間 (秒)', fontsize=8)
        ax1.set_ylabel('振幅', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=7)

        ax2 = self._fig.add_subplot(212)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz',
                                 ax=ax2, cmap='magma')
        ax2.set_title('スペクトログラム', fontsize=9)
        ax2.set_xlabel('時間 (秒)', fontsize=8)
        ax2.set_ylabel('周波数 (Hz)', fontsize=8)
        ax2.tick_params(labelsize=7)

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()

        self.an_status.config(text=f"分析完了 — スコア: {s}", fg='#27ae60')


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
