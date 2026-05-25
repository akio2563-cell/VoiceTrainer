import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import librosa

SAMPLE_RATE = 22050


class AudioEngine:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self._recorded_chunks = []
        self._stream = None
        self.is_recording = False

    def get_devices(self):
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                result.append({
                    'id': i,
                    'name': d['name'],
                    'channels': d['max_input_channels'],
                })
        return result

    def get_default_input_id(self):
        try:
            default = sd.query_devices(kind='input')
            for i, d in enumerate(sd.query_devices()):
                if d['name'] == default['name']:
                    return i
        except Exception:
            return None

    def start_recording(self, device_id=None):
        self._recorded_chunks = []
        self.is_recording = True

        def callback(indata, frames, time_info, status):
            if self.is_recording:
                self._recorded_chunks.append(indata.copy())

        self._stream = sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=self.sample_rate,
            dtype='float32',
            callback=callback,
        )
        self._stream.start()

    def stop_recording(self):
        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._recorded_chunks:
            return np.array([], dtype='float32')

        audio = np.concatenate(self._recorded_chunks, axis=0).flatten()
        return audio

    def save_wav(self, audio, filepath):
        int16 = np.clip(audio, -1.0, 1.0)
        int16 = (int16 * 32767).astype(np.int16)
        wavfile.write(filepath, self.sample_rate, int16)

    def load_wav(self, filepath):
        audio, _ = librosa.load(filepath, sr=self.sample_rate, mono=True)
        return audio
