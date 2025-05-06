import torch
from GRRNN import GrnnNet             # модель из репозитория
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor

# 1) Параметры
MODEL_PATH = 'model/GRRNN_WriterIdentification_dataset_CERUG-EN_model_vertical_aug_16-model_epoch_49.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) Инициализация и загрузка весов
model = GrnnNet(1, num_classes=105, mode='vertical').to(DEVICE)   # 105 – число авторов в CERUG-EN :contentReference[oaicite:6]{index=6}
state = torch.load(MODEL_PATH, map_location=DEVICE)              # загружаем state_dict :contentReference[oaicite:7]{index=7}
model.load_state_dict(state)                                     # заполняем модель параметрами :contentReference[oaicite:8]{index=8}
model.eval()                                                     # режим оценки для корректного инференса :contentReference[oaicite:9]{index=9}

# 3) Функция предобработки
def preprocess(img_path):
    img = Image.open(img_path).convert('L')          # в оттенки серого
    img = img.resize((64,128), resample=Image.BICUBIC)
    tensor = ToTensor()(np.array(img)).unsqueeze(0)  # [1,1,64,128]
    return tensor.to(DEVICE)

# 4) Функция инференса
def predict(image_path):
    x = preprocess(image_path)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()           # топ-1 предсказание
    return pred

if __name__ == '__main__':
    sample = 'inputM.png'
    author_id = predict(sample)
    print(f'Предсказанный ID автора: {author_id}')
