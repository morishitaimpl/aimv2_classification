import sys, os, pathlib, timm, time
sys.dont_write_bytecode = True
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
# from torchvision import models as tv_models
from transformers import AutoProcessor, Aimv2VisionModel
# from aim.v2 import mixins
# from aim.v2.torch import layers
# from aim.v1.torch import models as aim_models

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # HFの並列トークナイザ無効
torch.set_num_threads(1)   
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 学習回数
epochSize = 50

# 学習時のバッチのサイズ
batchSize = 4

cellSize = 500

classesSize = 3

# 学習時のサンプルを学習：検証データに分ける学習側の割合
splitRateTrain = 0.8

# データ変換
data_transforms = T.Compose([
    T.Resize(int(cellSize * 1.2)),
    T.RandomRotation(degrees = 15),
    T.RandomApply([T.GaussianBlur(5, sigma = (0.1, 5.0))], p = 0.5),
    T.ColorJitter(brightness = 0, contrast = 0, saturation = 0, hue = [-0.2, 0.2]),
    T.RandomHorizontalFlip(0.5),
    T.CenterCrop(cellSize),
    T.ToTensor(),
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

def calc_acc(output, label): # 結果が一致するラベルの数をカウントする
    p_arg = torch.argmax(output, dim = 1)
    return torch.sum(label == p_arg)

class build_model(nn.Module):
    def __init__(self, classesSize):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224")
        self.model = Aimv2VisionModel.from_pretrained("apple/aimv2-large-patch14-224")  # 出力埋め込み次元は 1024

        self.feat_dim = self.model.config.hidden_size
        self.bn = nn.BatchNorm1d(self.feat_dim)   
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.feat_dim, classesSize)

    def encode(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            out = self.model(pixel_values=pixel_values)
        return out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state.mean(dim=1)

    def forward(self, images):
        feats = self.encode(images)                 
        x = self.bn(feats)
        x = self.dropout(x)
        logits = self.classifier(x)              
        return logits

if __name__ == "__main__":
    import os
    from torchinfo import summary
    from torchviz import make_dot

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    mdl = build_model(classesSize)

    print(mdl)
    summary(mdl, (batchSize, 3, cellSize, cellSize))
    
    x = torch.randn(batchSize, 3, cellSize, cellSize).to(DEVICE) # 適当な入力
    y = mdl(x) # その出力
    
    img = make_dot(y, params = dict(mdl.named_parameters())) # 計算グラフの表示
    img.format = "png"
    img.render("_model_graph") # グラフを画像に保存
    os.remove("_model_graph") # 拡張子無しのファイルもできるので個別に削除