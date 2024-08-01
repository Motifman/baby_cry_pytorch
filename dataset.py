import os
import numpy as np
import librosa
import re
import json


def get_all_files(base_dir):
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # 拡張子の重複を取り除く
            normalized_file = re.sub(r'\.(3gp|wav|caf)\.\1$', r'.\1', file)
            if normalized_file.endswith(('.3gp', '.wav', '.caf')):
                all_files.append(os.path.join(root, normalized_file))
    return all_files


def extract_info(filename):
    # ファイル名のパターンから情報を抽出
    # pattern = r'-(\d+)\.(\d+)-(\w)-(\d+)-(\w\w)\.'
    pattern = r'-(\w)-(\d+)-(\w\w)\.'
    match = re.search(pattern, filename)
    if match:
        gender_code = match.group(1)
        age_code = match.group(2)
        reason_code = match.group(3)

        # マッピング
        gender = 'male' if gender_code == 'm' else 'female'
        age_map = {
            '04': '0-4 weeks',
            '48': '4-8 weeks',
            '26': '2-6 months',
            '72': '7 months - 2 years',
            '22': 'more than 2 years'
        }
        reason_map = {
            'hu': 'hungry',
            'bu': 'needs burping',
            'bp': 'belly pain',
            'dc': 'discomfort',
            'ti': 'tired',
            'lo': 'lonely',
            'ch': 'cold/hot',
            'sc': 'scared',
            'dk': 'don\'t know'
        }

        age = age_map.get(age_code, 'unknown')
        reason = reason_map.get(reason_code, 'unknown')

        return gender, age, reason
    print("not match...")
    print(filename)
    return None, None, None


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)  # サンプリングレートを維持
    return audio, sr


if __name__ == '__main__':
    base_dir = 'donateacry-corpus/'
    all_files = get_all_files(base_dir)

    data = []
    labels = []

    for file_path in all_files:
        gender, age, reason = extract_info(file_path)
        if reason:  # ラベルが正しく取得できた場合のみ処理
            audio, sr = load_audio(file_path)
            data.append(audio)
            labels.append({
                'gender': gender,
                'age': age,
                'reason': reason,
                'file_path': file_path,
                'sampling_rate': sr
            })

    # numpy配列に変換
    data = np.array(data, dtype=object)  # 可変長のオーディオシグナルを許容するためobject型
    labels = np.array(labels)

    # 必要であれば、データセットをファイルに保存
    np.save('crying_baby_audio_data.npy', data)
    with open('crying_baby_labels.json', 'w') as f:
        json.dump(labels.tolist(), f, indent=4)
