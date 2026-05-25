import numpy as np
import librosa

SR = 22050
MIN_DURATION_SEC = 0.5


class FeatureEngine:
    def __init__(self, sample_rate=SR):
        self.sr = sample_rate
        self.n_mfcc = 20

    def extract(self, audio: np.ndarray) -> dict | None:
        """Extract voice quality features. Returns None if audio too short."""
        if len(audio) < self.sr * MIN_DURATION_SEC:
            return None

        feats = {}

        # MFCC — timbre / texture
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        feats['mfcc_mean'] = np.mean(mfcc, axis=1)
        feats['mfcc_std'] = np.std(mfcc, axis=1)

        # Spectral shape
        sc = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        feats['sc_mean'] = float(np.mean(sc))
        feats['sc_std'] = float(np.std(sc))

        sb = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
        feats['sb_mean'] = float(np.mean(sb))

        sr_feat = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        feats['sr_mean'] = float(np.mean(sr_feat))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        feats['zcr_mean'] = float(np.mean(zcr))
        feats['zcr_std'] = float(np.std(zcr))

        # Harmonic/noise ratio proxy
        harmonic, _ = librosa.effects.hpss(audio)
        total_energy = float(np.mean(np.abs(audio))) + 1e-8
        feats['harmonic_ratio'] = float(np.mean(np.abs(harmonic))) / total_energy

        # RMS energy stability
        rms = librosa.feature.rms(y=audio)[0]
        feats['rms_mean'] = float(np.mean(rms))
        feats['rms_std'] = float(np.std(rms))

        # Chroma (pitch class profile)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        feats['chroma_mean'] = np.mean(chroma, axis=1)  # shape (12,)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        feats['contrast_mean'] = np.mean(contrast, axis=1)  # shape (7,)

        # Pitch stability (voiced frames only)
        f0, voiced, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
        )
        voiced_f0 = f0[voiced] if voiced is not None else np.array([])
        if len(voiced_f0) > 1:
            feats['pitch_stability'] = 1.0 / (float(np.std(voiced_f0)) + 1.0)
            feats['voiced_ratio'] = float(np.mean(voiced))
        else:
            feats['pitch_stability'] = 0.0
            feats['voiced_ratio'] = 0.0

        return feats

    def to_vector(self, feats: dict) -> np.ndarray | None:
        if feats is None:
            return None
        parts = [
            feats['mfcc_mean'],
            feats['mfcc_std'],
            [feats['sc_mean'], feats['sc_std'], feats['sb_mean'],
             feats['sr_mean'], feats['zcr_mean'], feats['zcr_std'],
             feats['harmonic_ratio'], feats['rms_mean'], feats['rms_std']],
            feats['chroma_mean'],
            feats['contrast_mean'],
            [feats['pitch_stability'], feats['voiced_ratio']],
        ]
        return np.concatenate([np.atleast_1d(p) for p in parts]).astype(np.float32)

    def readable_scores(self, feats: dict) -> dict:
        """Human-readable 0-100 scores for each voice quality dimension."""
        if feats is None:
            return {}

        scores = {}

        # 倍音の豊かさ: harmonic energy vs total
        scores['倍音の豊かさ'] = min(100.0, feats['harmonic_ratio'] * 120.0)

        # 声の安定性: spectral centroid stability (timbre consistency over time)
        # coefficient of variation of sc = sc_std / sc_mean → lower = more stable
        sc_cv = feats['sc_std'] / (feats['sc_mean'] + 1e-8)
        scores['声の安定性'] = min(100.0, max(0.0, 100.0 - sc_cv * 300.0))

        # 音の明るさ: spectral centroid normalized to 500-4500 Hz
        sc = feats['sc_mean']
        scores['音の明るさ'] = min(100.0, max(0.0, (sc - 500.0) / 4000.0 * 100.0))

        # 声の明瞭さ: voiced frame ratio
        scores['声の明瞭さ'] = min(100.0, feats['voiced_ratio'] * 100.0)

        # 響き: combo of harmonic ratio and spectral contrast
        resonance = feats['harmonic_ratio'] * 60.0 + float(np.mean(feats['contrast_mean'])) * 4.0
        scores['響き'] = min(100.0, max(0.0, resonance))

        return scores
