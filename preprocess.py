import numpy as np
import librosa
import librosa.display
import pandas as pd
import os
from tqdm import tqdm


def preprocess_audio(waveform, sr, target_sr=16000, max_length=112000):
    # サンプリングレートのリサンプリング
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    # ゼロパディングまたはトリミングで長さを統一
    if len(waveform) > max_length:
        waveform = waveform[:max_length]
    else:
        waveform = np.pad(waveform, (0, max_length - len(waveform)), mode='constant')

    # メルスペクトログラムに変換
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=target_sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 0～1の範囲に正規化
    mel_spec_db_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

    return mel_spec_db_normalized, waveform


def preprocess_and_save(waveforms, metadata, output_dir='processed_data'):
    os.makedirs(output_dir, exist_ok=True)

    processed_mel = []
    processed_wave = []

    for i, waveform in tqdm(enumerate(waveforms)):
        sr = metadata.loc[i, 'sample_rate']
        reason = metadata.loc[i, 'reason']
        filename = f"{reason}_{i}.npy"

        mel_spec_normalized, waveform = preprocess_audio(waveform, sr)
        processed_mel.append(mel_spec_normalized)
        processed_wave.append(waveform)

        np.save(os.path.join(output_dir, filename), mel_spec_normalized)

    return np.array(processed_mel), np.array(processed_wave)


# メイン処理
def main():
    waveforms = np.load('baby_cry_waveforms.npy', allow_pickle=True)
    metadata = pd.read_csv('baby_cry_metadata.csv')

    processed_mel, processed_wave = preprocess_and_save(waveforms, metadata)
    np.save('baby_cry_preprocessed_mel.npy', processed_mel)
    np.save('baby_cry_preprocessed_wave.npy', processed_wave)
    print("Preprocessing completed. Processed data saved.")


if __name__ == '__main__':
    main()
