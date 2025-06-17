import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import time

# --- 設定項目 ---
DATA_DIR = '/home/is/akihiro-i/datasets/ham10000'  # データセットのパスに変更してください
IMAGE_DIR_PART1 = os.path.join(DATA_DIR, 'HAM10000_images_part_1')
IMAGE_DIR_PART2 = os.path.join(DATA_DIR, 'HAM10000_images_part_2')
METADATA_FILE = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')

INPUT_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS_FOR_TIMING = 1 # 時間計測のためのエポック数
NUM_WORKERS = 4 # データローダーのワーカー数 (環境に合わせて調整)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# --- 1. データセットの定義 ---
class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir1, img_dir2, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.transform = transform
        # クラスラベルのマッピング (例: 'dx' 列を使用)
        self.classes = self.df['dx'].astype('category').cat.categories
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.df['label'] = self.df['dx'].map(self.class_to_idx)
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df['image_id'].iloc[idx] + '.jpg'
        # 画像がどちらのフォルダにあるか確認
        img_path_part1 = os.path.join(self.img_dir1, img_name)
        img_path_part2 = os.path.join(self.img_dir2, img_name)

        if os.path.exists(img_path_part1):
            image_path = img_path_part1
        elif os.path.exists(img_path_part2):
            image_path = img_path_part2
        else:
            raise FileNotFoundError(f"Image {img_name} not found in either directory.")

        image = Image.open(image_path).convert('RGB')
        label = self.df['label'].iloc[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# --- 2. データ変換とデータローダー ---
# 標準的な前処理
data_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNetの平均と標準偏差
])

try:
    full_dataset = HAM10000Dataset(csv_file=METADATA_FILE,
                                    img_dir1=IMAGE_DIR_PART1,
                                    img_dir2=IMAGE_DIR_PART2,
                                    transform=data_transforms)
except FileNotFoundError as e:
    print(f"エラー: データセットのファイルが見つかりません。パスを確認してください: {e}")
    exit()
except Exception as e:
    print(f"データセットのロード中にエラーが発生しました: {e}")
    exit()


# 時間計測のため、データセットの一部だけを使うこともできます (オプション)
# from torch.utils.data import Subset
# timing_dataset_size = BATCH_SIZE * 10 # 例: 10バッチ分
# if len(full_dataset) > timing_dataset_size:
#     indices = torch.randperm(len(full_dataset))[:timing_dataset_size]
#     timing_dataset = Subset(full_dataset, indices)
# else:
#     timing_dataset = full_dataset
# train_loader = DataLoader(timing_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
num_classes = full_dataset.num_classes

# --- 3. モデルの準備 ---
# model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1) # または weights=None で未学習モデル
model = models.resnet101(weights=None) # 未学習のResNet101を使用
# ResNetの最終層をHAM10000のクラス数に合わせて変更
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(DEVICE)

# --- 4. 損失関数とオプティマイザ ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # 学習率はお好みで

# --- 5. 学習ループと時間計測 ---
model.train() # 学習モード

total_iterations = 0
iteration_times = []
warmup_iterations = 5 # 最初の数イテレーションはCUDAの初期化などで不安定なため除外

print(f"\n--- 時間計測開始 (最初の {warmup_iterations} イテレーションはウォーミングアップ) ---")

for epoch in range(NUM_EPOCHS_FOR_TIMING):
    epoch_start_time = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        iteration_start_time = time.time() # CPU時間ベースの開始時刻

        # CUDAイベントによるGPU処理時間計測 (より正確)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        start_event.record() # GPU処理開始を記録

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        end_event.record() # GPU処理終了を記録
        torch.cuda.synchronize() # GPU処理の完了を待つ

        # GPU処理時間をミリ秒で取得
        gpu_time_ms = start_event.elapsed_time(end_event)

        iteration_end_time = time.time() # CPU時間ベースの終了時刻
        cpu_wall_time_ms = (iteration_end_time - iteration_start_time) * 1000

        total_iterations += 1

        if total_iterations > warmup_iterations:
            iteration_times.append(gpu_time_ms) # GPU時間を記録
            # iteration_times.append(cpu_wall_time_ms) # CPUウォールタイムを記録したい場合

        if (i + 1) % 10 == 0 or i == 0: # 10イテレーションごと、または最初のイテレーションで進捗表示
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS_FOR_TIMING}], Batch [{i+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, GPU Time: {gpu_time_ms:.2f} ms, CPU Wall Time: {cpu_wall_time_ms:.2f} ms")

        # 時間計測のために早く終了する場合 (オプション)
        # if total_iterations >= warmup_iterations + 20: # ウォームアップ後20イテレーション計測したら終了
        #     break
    # if total_iterations >= warmup_iterations + 20:
    #     break

    epoch_end_time = time.time()
    print(f"Epoch {epoch+1} completed in {(epoch_end_time - epoch_start_time):.2f} seconds")


# --- 6. 結果の集計と表示 ---
if iteration_times:
    avg_iteration_time_ms = sum(iteration_times) / len(iteration_times)
    print(f"\n--- 時間計測結果 ---")
    print(f"計測対象イテレーション数: {len(iteration_times)}")
    print(f"平均イテレーション処理時間 (GPU): {avg_iteration_time_ms:.2f} ms")

    # 1秒あたりの処理バッチ数 (スループットの目安)
    throughput_batches_per_sec = 1000.0 / avg_iteration_time_ms
    print(f"スループット (GPU時間ベース): {throughput_batches_per_sec:.2f} batches/sec")
    # 1秒あたりの処理画像数
    throughput_images_per_sec = throughput_batches_per_sec * BATCH_SIZE
    print(f"スループット (GPU時間ベース): {throughput_images_per_sec:.2f} images/sec")
else:
    print("\n計測データがありません。ウォーミングアップイテレーション数や計測エポック数を確認してください。")

print("\nスクリプト終了")