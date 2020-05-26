import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext


torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class Embedder(nn.Module):
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()
        
        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True)
        
    def forward(self, x):
        x_vec = self.embeddings(x)
        return x_vec


from utils.dataloader import get_IMDb_DataLoaders_and_TEXT
train_dl, val_dl, test_dl, TEXT = get_IMDb_DataLoaders_and_TEXT(
    max_length=256, batch_size=24)


batch = next(iter(train_dl))


net1 = Embedder(TEXT.vocab.vectors)


# [0]にする理由は0個目には, [0]にID列, [1]にそれぞれのベクトルの長さ(<pad>を除いたもの)が格納されているから
x = batch.Text[0]


batch.Text


x1 = net1(x)


x1.shape


# PositionalEncoder
# self-attentionが位置情報を扱えない
# 今回の実装では、文章中の単語の位置情報だけでなく、単語の分散表現それぞれの次元の位置情報も足している。
# 本来は、単語の分散表現内のそれぞれの要素の位置までは足す必要がない
class PositionalEncoder(nn.Module):
    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)
        
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / d_model)))
        
        # 0番目に新たな次元を追加する関数
        # ミニバッチ次元となる次元を足す
        # こうすることでxのミニバッチサイズがいくつであろうと、この次元に沿ってブロードキャストされる
        self.pe = pe.unsqueeze(0)
        
        # 勾配を計算しないようにする
        self.pe.requires_grad = False
        
    def forward(self, x):
        # ただ x と足すだけでも良いが、xの値が小さいので、xの値を定数倍してからpositional encodingしている
        ret = math.sqrt(self.d_model) * x + self.pe
        return ret


net1 = Embedder(TEXT.vocab.vectors)
net2 = PositionalEncoder()

x = batch.Text[0]
x1 = net1(x)
x2 = net2(x1)

x2.shape


class Attention(nn.Module):
    def __init__(self, d_model=300):
        super().__init__()
        
        # nn.Linear は全結合層は作る関数
        # nn.Linear(in_features, out_features, bias=True)
        # in_features, out_features は全結合層の入力と出力の次元数
        # 特徴量を変換するための層
        # query, key, valueはそれぞれ同じ特徴量から作られるが、違う重みを通して特徴量を変換する
        # self.q_linear(input)とやると、返り値が出力になる
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        # 出力に用いる全結合層
        self.out = nn.Linear(d_model, d_model)
        
        # Attentionの大きさ調整の変数
        # 別のメソッドで使うから
        self.d_k = d_model
        
    # 入力は ミニバッチx１文の単語数(256)x分散表現の次元(300)
    def forward(self, q, k, v, mask):
        # 全結合層で特徴量を変換
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        
        # torch.matmul(input, other, out=None) → Tensor
        # inputとotherの行列
        # k.transpose(dim0, dim1)はdim0とdim1を転置したTensorを返す
        # 0次元はミニバッチサイズであるため1次元目と2次元目を転置する
        # / math.sqrt(self.d_k) は matmulすると大きくなりすぎるので正規化をしている
        # 行列の形は ミニバッチx256x300 -> ミニバッチx256x256
        # 出力の256x256にはそれぞれの単語に対するそれぞれの単語の関連度がスカラで格納されている
        # この時点で片方でも<pad>の場合はその関連度の値は0になる
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        
        # maskを計算
        # <pad>をattentin weightsが0となるようにマスクをする
        # softmaxするため -無限 にすることで attention weightsが0となる
        mask = mask.unsqueeze(1)
        # tensor.masked_fill(mask, value) maskがTrueとなるところを、valueで置き換える
        # 今回だと<pad>の分散表現の値がすべて0であるため、計算後の値が0になっているのでそこを-1e9に置き換えている
        weights = weights.masked_fill(mask == 0, -1e9)
        
        # torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
        # dim=-1は最後尾の次元を表しているので、ある単語に対してそれぞれの単語との関連度が入っているベクトルでsoftmaxをしている
        # テンソルの形は変化なし
        normlized_weights = F.softmax(weights, dim=-1)
        
        # ミニバッチを省略 256x256 ✕ 256x300 = 256x300
        output = torch.matmul(normlized_weights, v)
        output = self.out(output)
        
        # normlized_weightsを返すのは後で確認を行うため、
        return output, normlized_weights
        


# 全結合層2つとdropoutで特徴量を変換するだけの層
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
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
        # LayerNormalization
        # 単語ごとの、300個の要素について、平均が0,標準偏差が1になるように正規化を行う。
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        self.attn = Attention(d_model)
        
        # Attentionのあとの全結合層2つ
        self.ff = FeedForward(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
        
    def forward(self, x, mask):
        #######################################
        x_normlized = self.norm_1(x)
        # ここで q, k, vですべてx_normlizedなのがself-attention
        output, normlized_weights = self.attn(
        x_normlized, x_normlized, x_normlized, mask )
        
        x2 = x + self.dropout_1(output)
        ########################################
        # ここまでが一ブロック
        # まずは、LayerNormで入力を正規化
        # 正規化した入力でself-attention
        # その出力と正規化していない入力xを足し合わせる 残差ネットワーク
        # 残差ネットワーク (ResNet) ある層で求める最適な出力を学習するのではなく、層の入力を参照した残差関数を学習する
        # つまり入力との差分のみを学習するようにする -> そのために入力をそのまま加算したものを出力とする
        #
        # そしてこれを2回繰り返す
        
        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))
        
        return output, normlized_weights


net1 = Embedder(TEXT.vocab.vectors)
net2 = PositionalEncoder(d_model=300, max_seq_len=256)
net3 = TransformerBlock(d_model=300)

x = batch.Text[0]
# <pad>の単語IDが1より
input_pad = 1
# False==0 が True になる
input_mask = (x get_ipython().getoutput("= input_pad)")

input_mask.shape
input_mask.unsqueeze(1).shape


x1 = net1(x)
x2 = net2(x1)
x3, normlized_weights = net3(x2, input_mask)

print(x3.shape)


class ClassificationHead(nn.Module):
    def __init__(self, d_model=300, output_dim=2):
        super().__init__()
        
        self.linear = nn.Linear(d_model, output_dim)
        
        # torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
        # mean 正規分布の平均   std 正規分布の標準偏差
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        
    def forward(self, x):
        # 先頭の単語<cls>を取り出す
        x0 = x[:, 0, :]
        out = self.linear(x0)
        
        return out


class TransformerClassification(nn.Module):
    def __init__(self, text_embedding_vectors, d_model=300, max_seq_len=256, output_dim=2):
        super().__init__()
        
        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(d_model=d_model)
        self.net3_2 = TransformerBlock(d_model=d_model)
        self.net4 = ClassificationHead(output_dim=output_dim, d_model=d_model)
        
    def forward(self, x, mask):
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3_1, normlized_weights_1 = self.net3_1(x2, mask)
        x3_2, normlized_weights_2 = self.net3_1(x3_1, mask)
        x4 = self.net4(x3_2)
        return x4, normlized_weights_1, normlized_weights_2


batch = next(iter(train_dl))

net = TransformerClassification(
    text_embedding_vectors=TEXT.vocab.vectors, d_model=300, max_seq_len=256,
    output_dim=2)

x = batch.Text[0]
input_mask = (x get_ipython().getoutput("= input_pad)")
out, normlized_weights_1, normlized_weights_2 = net(x, input_mask)

print(out.shape)
print(F.softmax(out, dim=1))



