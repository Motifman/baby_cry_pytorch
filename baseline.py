import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import seaborn as sns
from argparse import ArgumentParser


class BabyCryDataset(Dataset):
    def __init__(self, waveforms_path, mel_specs_path, metadata_path):
        self.waveforms_path = waveforms_path
        self.mel_specs_path = mel_specs_path
        self.metadata = pd.read_csv(metadata_path)

        self.waveforms_np = np.load(waveforms_path)
        self.mel_specs_np = np.load(mel_specs_path)

        self.labels = self.metadata['reason'].tolist()
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.metadata['reason'].unique()))}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        waveform = self.waveforms_np[idx]
        mel_spec = self.mel_specs_np[idx]
        label = self.label_map[self.labels[idx]]

        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
        mel_spec_tensor = torch.tensor(mel_spec, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return waveform_tensor, mel_spec_tensor, label_tensor


def set_seed(seed=123):
    print("set seed numpy and torch")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    np.random.seed(seed)


class CNN_baseline(nn.Module):
    def __init__(self, H, W, num_classes):
        super().__init__()
        self.H = H  # 128
        self.W = W  # 219

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 7), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 7), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 5), stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)

        x = self.bn2(self.conv2(x))
        x = self.relu(x)

        x = self.bn3(self.conv3(x))
        x = self.relu(x)

        x = self.bn4(self.conv4(x))
        x = self.relu(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        y = self.fc2(x)

        return y


class EarlyStopping:
    """earlystopping"""

    def __init__(self, patience=3, verbose=False, path="checkpoint_model.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss, model):
        if np.isnan(val_loss):
            self.early_stop = True
            print("Early stopping due to NaN loss.")
            score = -99999
        else:
            score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.checkpoint(val_loss, model)
            self.counter = 0

    def checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def plot_label_count(labels):
    label_count = Counter(labels)
    labels, values = zip(*label_count.items())

    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(labels)))
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=colors)

    plt.xlabel('label')
    plt.ylabel('number')

    plt.show()


def accuracy(outputs, targets):
    # outputs: (B, D)
    # targets: (B,)
    outputs = torch.argmax(outputs, dim=1)  # (B, T)
    return (outputs == targets).float().mean()


def train(model, optimizer, criterion, train_loader, device):
    sum_loss = 0
    sum_acc = 0

    model.train()
    for _, mels, labels in tqdm(train_loader):
        mels = mels.to(device)
        mels = mels.unsqueeze(1)
        labels = labels.to(device)

        outputs = model(mels)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        sum_acc += accuracy(outputs, labels).item()

    return sum_loss / len(train_loader), sum_acc / len(train_loader)


def eval(model, criterion, eval_loader, device):
    sum_loss = 0
    sum_acc = 0
    pred_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for _, mels, labels in tqdm(eval_loader):
            mels = mels.to(device)
            mels = mels.unsqueeze(1)
            labels = labels.to(device)

            outputs = model(mels)
            loss = criterion(outputs, labels)

            sum_loss += loss.item()
            sum_acc += accuracy(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)
            pred_list = pred_list + preds.cpu().tolist()
            labels_list = labels_list + labels.cpu().tolist()

    return sum_loss / len(eval_loader), sum_acc / len(eval_loader), pred_list, labels_list


def evaluate_best_model(model, criterion, train_loader, eval_loader, device):
    train_loss, train_metric, pred_list1, labels_list1 = eval(model, criterion, train_loader, device)
    eval_loss, eval_metric, pred_list2, labels_list2 = eval(model, criterion, eval_loader, device)
    pred_list = pred_list1 + pred_list2
    labels_list = labels_list1 + labels_list2
    return train_loss, train_metric, eval_loss, eval_metric, pred_list, labels_list


def plot_metrics(metrics, path1, path2):
    with plt.style.context("ggplot"):
        fig1 = plt.figure()
        plt.plot(metrics["epoch"], metrics["train_loss"], label="train_loss")
        plt.plot(metrics["epoch"], metrics["eval_loss"], label="eval_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(path1)
        plt.close()

        fig2 = plt.figure()
        plt.plot(metrics["epoch"], metrics["train_acc"], label="train_acc")
        plt.plot(metrics["epoch"], metrics["eval_acc"], label="eval_acc")
        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.savefig(path2)
        plt.close()


def main():
    # argparse
    parser = ArgumentParser("cnn_baby_cry")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--class_num", type=int, default=9)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weighted", type=bool, default=False)
    args = parser.parse_args()

    # hyper parameters
    waveforms_path = 'baby_cry_preprocessed_wave.npy'
    mel_specs_path = 'baby_cry_preprocessed_mel.npy'
    metadata_path = 'baby_cry_metadata.csv'
    batch_size = args.batch_size
    class_num = args.class_num
    seed = args.seed
    num_epoch = args.num_epoch
    lr = args.lr
    optim = args.optim
    patience = args.patience
    weighted = args.weighted
    shuffle = True

    # seed
    set_seed(seed)

    # メタデータとラベルを取得
    metadata_df = pd.read_csv(metadata_path)
    labels = metadata_df['reason']

    # TrainとValidationに分割（クラス比率を維持）
    print(len(labels))
    train_indices, val_indices = train_test_split(
        metadata_df.index,
        test_size=0.2,  # 検証データの割合を指定
        stratify=labels,  # クラス比率を維持
        random_state=seed  # 再現性のためのランダムシード
    )

    # Datasetクラスのインスタンスを作成
    train_dataset = BabyCryDataset(waveforms_path, mel_specs_path, metadata_path)
    val_dataset = BabyCryDataset(waveforms_path, mel_specs_path, metadata_path)

    labels = train_dataset.labels

    # クラスごとのサンプル数をカウント
    class_counts = Counter(labels)

    # クラスのインデックス順に並べる
    classes = sorted(class_counts.keys())

    # クラスごとの重みを計算 (サンプル数の逆数)
    weights = [1.0 / class_counts[cls] for cls in classes]

    # 重みを正規化して合計が1になるようにする（オプション）
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()

    # インデックスを基にデータセットを分割
    train_dataset.waveforms_np = train_dataset.waveforms_np[train_indices]
    train_dataset.mel_specs_np = train_dataset.mel_specs_np[train_indices]
    train_dataset.metadata = train_dataset.metadata.iloc[train_indices].reset_index(drop=True)
    train_dataset.labels = train_dataset.metadata['reason'].tolist()

    val_dataset.waveforms_np = val_dataset.waveforms_np[val_indices]
    val_dataset.mel_specs_np = val_dataset.mel_specs_np[val_indices]
    val_dataset.metadata = val_dataset.metadata.iloc[val_indices].reset_index(drop=True)
    val_dataset.labels = val_dataset.metadata['reason'].tolist()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    eval_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    # labelの種類を分析
    # 辞書にラベルごとの個数を格納する
    for waveforms, mel_specs, labels in train_loader:
        print("Batch of waveforms:", waveforms.size())
        print("Batch of mel spectrograms:", mel_specs.size())
        print("Batch of labels:", labels.size())
        _, H, W = mel_specs.shape
        break

    print(f"train_size: {len(train_loader)}, eval_size: {len(eval_loader)}")

    # labelごとの個数をプロット
    plot_label_count(train_dataset.labels)
    plot_label_count(val_dataset.labels)

    # model, optimizer, loss
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else r"cpu")

    model = CNN_baseline(H, W, class_num)
    model = model.to(device)
    if optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(optim)

    if weighted:
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    earlystopping = EarlyStopping(
        patience=patience,
        verbose=False,
        path=f"log/checkpoint_model.pth",
    )

    # metrics
    metrics = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "eval_loss": [],
        "eval_acc": [],
    }

    # train, eval
    for epoch in range(num_epoch):
        train_loss, train_acc = train(model, optimizer, criterion, train_loader, device)
        eval_loss, eval_acc, _, _ = eval(model, criterion, eval_loader, device)

        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["eval_loss"].append(eval_loss)
        metrics["eval_acc"].append(eval_acc)

        earlystopping(eval_loss, model=model)
        if earlystopping.early_stop:
            print(
                f"Early Stopping! best eval loss is {earlystopping.best_score}"
            )
            break

        print(f"Epoch={epoch}, train_loss={train_loss}, train_metric={train_acc}")
        print(f"Epoch={epoch}, eval_loss={eval_loss}, eval_metric={eval_acc}")

    # plot
    plot_metrics(metrics, "log/loss_curve.pdf", "log/acc_curve.pdf")

    # evaluate the best model
    print("evaluate best model")
    path = "log/checkpoint_model.pth"
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        train_loss, train_metric, eval_loss, eval_metric, pred_list, labels_list = evaluate_best_model(
            model, criterion,  train_loader, eval_loader, device
        )
    print(f"best_train_loss={train_loss}, best_train_metric={train_metric}")
    print(f"best_eval_loss={eval_loss}, best_eval_metric={eval_metric}")
    cm = confusion_matrix(labels_list, pred_list)
    sns.heatmap(cm)
    plt.savefig("log/confusion_matrix.pdf")


if __name__ == '__main__':
    main()