import librosa


if __name__ == '__main__':
    y, sr = librosa.load("./donateacry-corpus/donateacry_corpus_cleaned_and_updated_data/belly_pain/549a46d8-9c84-430e-ade8-97eae2bef787-1430130772174-1.7-m-48-bp.wav")

    print(f"sr:{sr}")
    print(f"y.shape:{y.shape}")
    print(f"y:{y}")