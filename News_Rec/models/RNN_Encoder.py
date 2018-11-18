from models.basemodel import basemodel
# 压缩稀疏矩阵的行
from scipy.sparse import csr_matrix
# 对多维数组对象的支持 支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
import numpy as np
# 引入张量运算等 深度学习框架（利用variable和function构建计算图）
import torch
import random
# 引入autograd中的变量 自动求导数工具包
from torch.autograd import Variable
# 构建神经网络
import torch.nn.functional as F
#词表中词的个数96566个
VOCAB_SIZE = 96566
# 设置每个词向量的维度大小为10
EMBEDDING_SIZE = 10
# 继承torch.nn的模块
class RNN_Encoder(torch.nn.Module):
    # 初始化模块
    def __init__(self, n_feature, n_hidden, vocab_size, embedding_dim, user_size, item_size):
        super(RNN_Encoder, self).__init__() #对RNN模块进行初始化
        self.user_embedding = torch.nn.Embedding(user_size, embedding_dim) #
        self.item_embedding = torch.nn.Embedding(item_size, embedding_dim)
        self.wrd_embedding = torch.nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.rnn = torch.nn.LSTM(
            input_size=n_feature,
            hidden_size=n_hidden,
            num_layers=1,
            dropout=0.2,
            batch_first=True
        )

    def forward(self, users, items, item_ctx, item_len):
        embedded_user = self.user_embedding(users)
        embedded_item = self.item_embedding(items)
        embedded_ctx = self.wrd_embedding(item_ctx)

        r_out, h = self.rnn(embedded_ctx)
        item_encode = []
        for i in range(len(item_len)):
            last_pos = item_len[i].data[0] - 1
            item_encode.append(r_out[i, last_pos, :])

        ctx_feature = torch.stack(item_encode, dim=1)
        pred = torch.sum(embedded_user.mul(ctx_feature), dim=1)
        pred += torch.sum(embedded_item.mul(embedded_user), dim=1)
        return pred

class MF_RNN(basemodel):
    """基于项目内容的K近邻方法，计算相似项目使用新闻内容的tfidf值，而不采用评分
    计算该用户对的最近似K个项目的评分加权预测对该项目的评分
    """
    def __init__(self, user_news_df, news_ctx_df, epoch):
        basemodel.__init__(self, user_news_df)
        self.user_news_df = user_news_df
        self.news_ctx_df = news_ctx_df
        self.rnn_encoder = RNN_Encoder(EMBEDDING_SIZE, EMBEDDING_SIZE, VOCAB_SIZE, EMBEDDING_SIZE, self.USER_NUM, self.ITEM_NUM)
        self.batch_size = 32
        self.epochs = epoch
        self.lr = 0.002
        self.opt = torch.optim.Adamax(self.rnn_encoder.parameters(), lr=self.lr)

    def train(self):
        train_len = self.user_news_df.shape[0]
        for epoch in range(self.epochs):
            for i in range(0, 1):#train_len // self.batch_size):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                end = end if end < train_len else train_len
                users = self.user_news_df.iloc[start: end, 0]
                items = self.user_news_df.iloc[start: end, 1]


                loss = self.update(users, items, self.batch_size)
                if i % 1 == 0:
                    print("Epoch %d Step %d / %d, loss: %g" % (epoch, i, train_len // self.batch_size, loss.data[0]))
            torch.save(self.rnn_encoder, 'trained_model/rnn_encoder.pkl')

    def update(self, users, items, batch):
        user_var = Variable(torch.from_numpy(np.array(users)))
        item_var = Variable(torch.from_numpy(np.array(items)))
        item_ids = []
        item_len = []
        maxlen = 0
        for item in items:
            ids = self.news_ctx_df.iloc[item, 1].split(' ')
            ids = list(map(int, ids))
            item_len.append(len(ids))
            maxlen = max(maxlen, len(ids))
            item_ids.append(ids)

        for i in range(0, len(item_ids)):
            for _ in range(maxlen - len(item_ids[i])):
                item_ids[i].append(0)

        itemctx_var = Variable(torch.from_numpy(np.array(item_ids)))
        item_len_var = Variable(torch.from_numpy(np.array(item_len)))
        pred = self.rnn_encoder(user_var, item_var, itemctx_var, item_len_var)

        target = Variable(torch.FloatTensor(np.array([1.0 for _ in range(batch)])))
        loss = F.mse_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss

    def update_pair(self, users, items, negitems, batch):
        user_var = Variable(torch.from_numpy(np.array(users)))
        item_var = Variable(torch.from_numpy(np.array(items)))
        neg_item_var = Variable(torch.from_numpy(np.array(negitems)))
        item_ids = []
        item_len = []
        maxlen = 0
        for item in items:
            ids = self.news_ctx_df.iloc[item, 1].split(' ')
            ids = list(map(int, ids))
            item_len.append(len(ids))
            maxlen = max(maxlen, len(ids))
            item_ids.append(ids)

        for i in range(0, len(item_ids)):
            for _ in range(maxlen - len(item_ids[i])):
                item_ids[i].append(0)

        neg_ids = []
        neg_len = []
        maxlen = 0
        for item in negitems:
            ids = self.news_ctx_df.iloc[item, 1].split(' ')
            ids = list(map(int, ids))
            neg_len.append(len(ids))
            maxlen = max(maxlen, len(ids))
            neg_ids.append(ids)

        for i in range(0, len(neg_ids)):
            for _ in range(maxlen - len(neg_ids[i])):
                neg_ids[i].append(0)

        itemctx_var = Variable(torch.from_numpy(np.array(item_ids)))
        item_len_var = Variable(torch.from_numpy(np.array(item_len)))
        negctx_var = Variable(torch.from_numpy(np.array(neg_ids)))
        neg_len_var = Variable(torch.from_numpy(np.array(neg_len)))

        pred_pos = self.rnn_encoder(user_var, item_var, itemctx_var, item_len_var)
        pred_neg = self.rnn_encoder(user_var, neg_item_var, negctx_var, neg_len_var)
        target = Variable(torch.FloatTensor(np.array([1.0 for _ in range(batch)])))
        pred = pred_pos - pred_neg
        loss = F.mse_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss

    def get_neg(self, users):
        neg_item = []
        for user in users:
            rand_item = random.randint(0, self.ITEM_NUM - 1)
            while self.ui_mat[user, rand_item] == 1:
                rand_item = random.randint(0, self.ITEM_NUM -1)
            neg_item.append(rand_item)
        return neg_item

    def predict(self, user, item):
        user_var = Variable(torch.from_numpy(np.array([user])))
        item_var = Variable(torch.from_numpy(np.array([item])))
        item_ids = self.news_ctx_df.iloc[item, 1].split(' ')
        itemctx_var = Variable(torch.from_numpy(np.array([item_ids], dtype='int64')))
        item_len = Variable(torch.from_numpy(np.array([len(item_ids)])))
        return self.rnn_encoder(user_var, item_var, itemctx_var, item_len)
