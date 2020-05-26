import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchtext


# Setup seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class Embedder(nn.Module):
    '''idで示されている単語をベクトルに変換します'''

    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True)
        # freeze=Trueによりバックプロパゲーションで更新されず変化しなくなります

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec



# 動作確認

# 前節のDataLoaderなどを取得
from utils.dataloader import get_IMDb_DataLoaders_and_TEXT
train_dl, val_dl, test_dl, TEXT = get_IMDb_DataLoaders_and_TEXT(
    max_length=256, batch_size=24)

# ミニバッチの用意
batch = next(iter(train_dl))

# モデル構築
net1 = Embedder(TEXT.vocab.vectors)

# 入出力
x = batch.Text[0]
x1 = net1(x)  # 単語をベクトルに

print("入力のテンソルサイズ：", x.shape)
print("出力のテンソルサイズ：", x1.shape)



class PositionalEncoder(nn.Module):
    '''入力された単語の位置を示すベクトル情報を付加する'''

    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()

        self.d_model = d_model  # 単語ベクトルの次元数

        # 単語の順番（pos）と埋め込みベクトルの次元の位置（i）によって一意に定まる値の表をpeとして作成
        pe = torch.zeros(max_seq_len, d_model)

        # GPUが使える場合はGPUへ送る、ここでは省略。実際に学習時には使用する
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                
                # 誤植修正_200510 #79
                # pe[pos, i + 1] = math.cos(pos /
                #                          (10000 ** ((2 * (i + 1))/d_model)))
                pe[pos, i + 1] = math.cos(pos /
                                          (10000 ** ((2 * i)/d_model)))

        # 表peの先頭に、ミニバッチ次元となる次元を足す
        self.pe = pe.unsqueeze(0)

        # 勾配を計算しないようにする
        self.pe.requires_grad = False

    def forward(self, x):

        # 入力xとPositonal Encodingを足し算する
        # xがpeよりも小さいので、大きくする
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret



# 動作確認

# モデル構築
net1 = Embedder(TEXT.vocab.vectors)
net2 = PositionalEncoder(d_model=300, max_seq_len=256)

# 入出力
x = batch.Text[0]
x1 = net1(x)  # 単語をベクトルに
x2 = net2(x1)

print("入力のテンソルサイズ：", x1.shape)
print("出力のテンソルサイズ：", x2.shape)



class Attention(nn.Module):
    '''Transformerは本当はマルチヘッドAttentionですが、
    分かりやすさを優先しシングルAttentionで実装します'''

    def __init__(self, d_model=300):
        super().__init__()

        # SAGANでは1dConvを使用したが、今回は全結合層で特徴量を変換する
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        # 出力時に使用する全結合層
        self.out = nn.Linear(d_model, d_model)

        # Attentionの大きさ調整の変数
        self.d_k = d_model

    def forward(self, q, k, v, mask):
        # 全結合層で特徴量を変換
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        # Attentionの値を計算する
        # 各値を足し算すると大きくなりすぎるので、root(d_k)で割って調整
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        # ここでmaskを計算
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)

        # softmaxで規格化をする
        normlized_weights = F.softmax(weights, dim=-1)

        # AttentionをValueとかけ算
        output = torch.matmul(normlized_weights, v)

        # 全結合層で特徴量を変換
        output = self.out(output)

        return output, normlized_weights



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        '''Attention層から出力を単純に全結合層2つで特徴量を変換するだけのユニットです'''
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # LayerNormalization層
        # https://pytorch.org/docs/stable/nn.html?highlight=layernorm
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Attention層
        self.attn = Attention(d_model)

        # Attentionのあとの全結合層2つ
        self.ff = FeedForward(d_model)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 正規化とAttention
        x_normlized = self.norm_1(x)
        output, normlized_weights = self.attn(
            x_normlized, x_normlized, x_normlized, mask)
        
        x2 = x + self.dropout_1(output)

        # 正規化と全結合層
        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output, normlized_weights



# 動作確認

# モデル構築
net1 = Embedder(TEXT.vocab.vectors)
net2 = PositionalEncoder(d_model=300, max_seq_len=256)
net3 = TransformerBlock(d_model=300)

# maskの作成
x = batch.Text[0]
input_pad = 1  # 単語のIDにおいて、'<pad>': 1 なので
input_mask = (x get_ipython().getoutput("= input_pad)")
print(input_mask[0])

# 入出力
x1 = net1(x)  # 単語をベクトルに
x2 = net2(x1)  # Positon情報を足し算
x3, normlized_weights = net3(x2, input_mask)  # Self-Attentionで特徴量を変換

print("入力のテンソルサイズ：", x2.shape)
print("出力のテンソルサイズ：", x3.shape)
print("Attentionのサイズ：", normlized_weights.shape)



class ClassificationHead(nn.Module):
    '''Transformer_Blockの出力を使用し、最後にクラス分類させる'''

    def __init__(self, d_model=300, output_dim=2):
        super().__init__()

        # 全結合層
        self.linear = nn.Linear(d_model, output_dim)  # output_dimはポジ・ネガの2つ

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]  # 各ミニバッチの各文の先頭の単語の特徴量（300次元）を取り出す
        out = self.linear(x0)

        return out



# 動作確認

# ミニバッチの用意
batch = next(iter(train_dl))

# モデル構築
net1 = Embedder(TEXT.vocab.vectors)
net2 = PositionalEncoder(d_model=300, max_seq_len=256)
net3 = TransformerBlock(d_model=300)
net4 = ClassificationHead(output_dim=2, d_model=300)

# 入出力
x = batch.Text[0]
x1 = net1(x)  # 単語をベクトルに
x2 = net2(x1)  # Positon情報を足し算
x3, normlized_weights = net3(x2, input_mask)  # Self-Attentionで特徴量を変換
x4 = net4(x3)  # 最終出力の0単語目を使用して、分類0-1のスカラーを出力

print("入力のテンソルサイズ：", x3.shape)
print("出力のテンソルサイズ：", x4.shape)



# 最終的なTransformerモデルのクラス


class TransformerClassification(nn.Module):
    '''Transformerでクラス分類させる'''

    def __init__(self, text_embedding_vectors, d_model=300, max_seq_len=256, output_dim=2):
        super().__init__()

        # モデル構築
        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(d_model=d_model)
        self.net3_2 = TransformerBlock(d_model=d_model)
        self.net4 = ClassificationHead(output_dim=output_dim, d_model=d_model)

    def forward(self, x, mask):
        x1 = self.net1(x)  # 単語をベクトルに
        x2 = self.net2(x1)  # Positon情報を足し算
        x3_1, normlized_weights_1 = self.net3_1(
            x2, mask)  # Self-Attentionで特徴量を変換
        x3_2, normlized_weights_2 = self.net3_2(
            x3_1, mask)  # Self-Attentionで特徴量を変換
        x4 = self.net4(x3_2)  # 最終出力の0単語目を使用して、分類0-1のスカラーを出力
        return x4, normlized_weights_1, normlized_weights_2



# 動作確認

# ミニバッチの用意
batch = next(iter(train_dl))

# モデル構築
net = TransformerClassification(
    text_embedding_vectors=TEXT.vocab.vectors, d_model=300, max_seq_len=256, output_dim=2)

# 入出力
x = batch.Text[0]
input_mask = (x get_ipython().getoutput("= input_pad)")
out, normlized_weights_1, normlized_weights_2 = net(x, input_mask)

print("出力のテンソルサイズ：", out.shape)
print("出力テンソルのsigmoid：", F.softmax(out, dim=1))

