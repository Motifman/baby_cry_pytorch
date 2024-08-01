import os
import numpy as np
import librosa
import pandas as pd


# def extract_metadata(filename):
#     # ファイル名からメタデータを抽出
#     base_name = os.path.splitext(filename)[0]
#     parts = base_name.split('-')
#
#     # ファイル名から抽出した各部分の意味
#     gender = parts[-3]
#     age = parts[-2]
#     reason = parts[-1]
#
#     return gender, age, reason


def extract_metadata(filename):
    # ファイル名から拡張子を取り除く
    base_name, ext = os.path.splitext(filename)
    if ".caf" in base_name:
        base_name, ext = os.path.splitext(base_name)
    parts = base_name.split('-')

    # ファイル名から抽出した各部分の意味
    if len(parts) >= 3:
        gender = parts[-3]
        age = parts[-2]
        reason = parts[-1]
    else:
        # parts の長さが不足している場合はデフォルト値を設定する
        gender = 'unknown'
        age = 'unknown'
        reason = 'unknown'

    return gender, age, reason


def load_audio(file_path):
    # ファイルを生波形として読み込む
    waveform, sr = librosa.load(file_path, sr=None)
    return waveform, sr


def process_directory(directory):
    data = []
    metadata = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.3gp', '.wav', '.caf', '.caf.caf')):
                file_path = os.path.join(root, file)
                try:
                    waveform, sr = load_audio(file_path)
                    len_wave = waveform.shape[0]
                    gender, age, reason = extract_metadata(file)

                    if len_wave > 10000:
                        data.append(waveform)
                        metadata.append({
                            'filename': file,
                            'gender': gender,
                            'age': age,
                            'reason': reason,
                            'sample_rate': sr
                        })
                    else:
                        print("too short...")
                        print(len_wave)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    return np.array(data, dtype=object), pd.DataFrame(metadata)


# メイン処理
def main():
    # 各ディレクトリのパスを指定
    directories = [
        'donateacry-corpus',
    ]

    all_data = []
    all_metadata = []

    for directory in directories:
        print(f"Processing directory: {directory}")
        data, metadata = process_directory(directory)
        all_data.append(data)
        all_metadata.append(metadata)

    # データとメタデータを統合
    all_data = np.concatenate(all_data, axis=0)
    all_metadata = pd.concat(all_metadata, ignore_index=True)

    # データを保存
    np.save('baby_cry_waveforms.npy', all_data)
    all_metadata.to_csv('baby_cry_metadata.csv', index=False)


if __name__ == '__main__':
    main()
