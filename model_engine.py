import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

MIN_SAMPLES = 3


class VoiceModel:
    def __init__(self):
        self.scaler = StandardScaler()
        # nu=0.05 means we allow 5% of training samples to be treated as outliers
        self.svm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
        self.is_trained = False
        self._vectors: list[np.ndarray] = []

    def add_sample(self, vector: np.ndarray):
        self._vectors.append(vector.astype(np.float64))

    @property
    def sample_count(self):
        return len(self._vectors)

    def train(self):
        if len(self._vectors) < MIN_SAMPLES:
            raise ValueError(f"最低{MIN_SAMPLES}サンプル必要です（現在: {len(self._vectors)}）")

        X = np.array(self._vectors, dtype=np.float64)
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled)
        self.is_trained = True

    def predict(self, vector: np.ndarray) -> tuple[bool, float]:
        """Returns (is_good, score_0_to_100)."""
        if not self.is_trained:
            raise RuntimeError("モデルが未学習です")

        X = vector.reshape(1, -1).astype(np.float64)
        X_scaled = self.scaler.transform(X)
        decision = float(self.svm.decision_function(X_scaled)[0])

        # decision > 0 → inlier (good), < 0 → outlier (bad)
        # Map to 0-100: center at 50 when decision=0
        score = float(np.clip(50.0 + decision * 30.0, 0.0, 100.0))
        return decision >= 0, score

    def save(self, filepath: str):
        joblib.dump({
            'scaler': self.scaler,
            'svm': self.svm,
            'is_trained': self.is_trained,
            'vectors': self._vectors,
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.svm = data['svm']
        self.is_trained = data['is_trained']
        self._vectors = data['vectors']
