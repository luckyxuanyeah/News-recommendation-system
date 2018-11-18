from models.basemodel import basemodel
from scipy.sparse import csr_matrix
import numpy as np
import torch
import random
from torch.autograd import Variable
import torch.nn.functional as F
import json


EMBEDDING_SIZE = 40
TOPIC_SIZE = 50

class model_MF(torch.nn.Module):
    def __init__(self, embedding_dim, topic_size,
                 user_size, item_size):
        super(model_MF, self).__init__()
        self.user_embedding = torch.nn.Embedding(user_size, embedding_dim)
        self.item_embedding = torch.nn.Embedding(item_size, embedding_dim)
        self.topic_embedding = Variable(torch.rand(topic_size, embedding_dim))

    def forward(self, users, items, item_ctx):
        embedded_user = self.user_embedding(users)
        embedded_item = self.item_embedding(items)
        embedded_ctx = torch.matmul(item_ctx, self.topic_embedding)
        pred = torch.sum(embedded_user.mul(embedded_ctx), dim=1)
        pred += torch.sum(embedded_item.mul(embedded_user), dim=1)
        return pred


class lda_MF(basemodel):
    """基于项目内容的K近邻方法，计算相似项目使用新闻内容的tfidf值，而不采用评分
    计算该用户对最近似K个项目的评分加权预测对该项目的评分
    """
    def __init__(self, user_news_df, epoch):
        basemodel.__init__(self, user_news_df)
        self.user_news_df = user_news_df
        self.model_mf = model_MF(EMBEDDING_SIZE, TOPIC_SIZE, self.USER_NUM, self.ITEM_NUM)
        self.batch_size = 100
        self.epochs = epoch
        self.lr = 0.02
        self.opt = torch.optim.Adamax(self.model_mf.parameters(), lr=self.lr)
        topic_file = open('Data/news_topic.txt', 'r', encoding='utf-8')
        self.news_topic = np.zeros((self.ITEM_NUM, TOPIC_SIZE))
        topic = json.loads(topic_file.readline())
        for i in range(self.ITEM_NUM):
            for tpc in topic[str(i)]:
                self.news_topic[i, tpc[0]] = tpc[1]


    def train(self, pairwise=False):
        train_len = self.user_news_df.shape[0]
        for epoch in range(self.epochs):
            for i in range(0, train_len // self.batch_size):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                end = end if end < train_len else train_len
                users = self.user_news_df.iloc[start: end, 0]
                items = self.user_news_df.iloc[start: end, 1]
                if pairwise == False:
                    loss = self.update(users, items, self.batch_size)
                else:
                    neg_items = self.get_neg(users)
                    loss = self.update_pair(users, items, neg_items, self.batch_size)
                if i % 100 == 0:
                    print("Epoch %d Step %d / %d, loss: %g" % (epoch, i, train_len // self.batch_size, loss.data[0]))
            torch.save(self.model_mf, 'trained_model/lda_mf.pkl')

    def update(self, users, items, batch):
        user_var = Variable(torch.from_numpy(np.array(users)))
        item_var = Variable(torch.from_numpy(np.array(items)))
        itemctx_var = Variable(torch.FloatTensor(np.array([self.news_topic[item] for item in items])))
        pred = self.model_mf(user_var, item_var, itemctx_var)

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
        itemctx_var = Variable(torch.FloatTensor(np.array([self.news_topic[item] for item in items])))
        negctx_var = Variable(torch.FloatTensor(np.array([self.news_topic[item] for item in negitems])))
        pred_pos = self.model_mf(user_var, item_var, itemctx_var)
        pred_neg = self.model_mf(user_var, neg_item_var, negctx_var)
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
        itemctx_var = Variable(torch.FloatTensor(self.news_topic[item]))
        return self.model_mf(user_var, item_var, itemctx_var).data[0]
