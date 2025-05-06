import os
import json
import torch
import torch.nn.functional as F
from GRRNN import GrnnNet
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor

# 1) Настройки
BACKBONE_WEIGHTS = 'model/GRRNN_WriterIdentification_dataset_CERUG-EN_model_vertical_aug_16-model_epoch_49.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 512           # размер скрытого состояния RNN в GRRNN
PROTO_DB = 'prototypes.json'

# 2) Сборка модели и загрузка весов
class EmbeddingModel(torch.nn.Module):
    def __init__(self, num_classes=105):
        super().__init__()
        # Загружаем GRRNN без головы классификации
        self.net = GrnnNet(1, num_classes=num_classes, mode='vertical')
        # classifier заменим на identity
        self.net.classifier = torch.nn.Identity()
    def forward(self, x):
        # вернёт тензор [batch, EMBED_DIM]
        return self.net(x)

model = EmbeddingModel().to(DEVICE)
state = torch.load(BACKBONE_WEIGHTS, map_location=DEVICE)
# удаляем из словаря ключи classifier.* при загрузке
state = {k: v for k,v in state.items() if not k.startswith('classifier')}
model.load_state_dict(state, strict=False)
model.eval()

# 3) Функции предобработки и эмбеддинга
def preprocess(path):
    img = Image.open(path).convert('L')
    img = img.resize((64,128), resample=Image.BICUBIC)
    tensor = ToTensor()(np.array(img)).unsqueeze(0).to(DEVICE)
    return tensor

@torch.no_grad()
def get_embedding(path):
    x = preprocess(path)
    emb = model(x)              # [1, EMBED_DIM]
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()[0]  # [EMBED_DIM]

# 4) Работа с базой прототипов
def load_prototypes():
    if os.path.exists(PROTO_DB):
        return json.load(open(PROTO_DB,'r'))
    return {}

def save_prototypes(db):
    json.dump(db, open(PROTO_DB,'w'), indent=2)

def register_person(name, sample_paths):
    """
    sample_paths: список путей к изображениям рукописи человека
    """
    embs = [get_embedding(p) for p in sample_paths]
    proto = np.mean(embs, axis=0).tolist()
    db = load_prototypes()
    db[name] = proto
    save_prototypes(db)
    print(f'Зарегистрирован: {name}')

def identify(path):
    """
    Возвращает имя наиболее похожего прототипа
    """
    db = load_prototypes()
    if not db:
        raise RuntimeError("Нет зарегистрированных пользователей")
    query = get_embedding(path)
    # Косинусное сходство
    sims = {}
    for name, proto in db.items():
        sims[name] = np.dot(query, proto) / (np.linalg.norm(proto) + 1e-8)
    # выбираем максимальную
    best = max(sims, key=sims.get)
    return best, sims[best]

# 5) Пример использования
if __name__ == '__main__':
    # -- Шаг 1: регистрируем двух людей
    register_person('Maksim', ['I1.png', 'I2.png', 'I3.png'])
    register_person('Mil`a',   ['M1.png',   'M2.png',   'M3.png'])

    # -- Шаг 2: идентифицируем нового образца
    name, score = identify('inputM.png')
    print(f'Это {name} (score={score:.3f})')
