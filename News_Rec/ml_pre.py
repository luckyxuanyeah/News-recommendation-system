# 强大的Python数据分析工具包
import pandas as pd
import random

def split_user():
    data_df = pd.read_csv("Data/rating.csv", sep=',', header=0)
    user_list = dict()
    for i in range(len(data_df)):
        user = data_df.iloc[i, 0] - 1
        item = data_df.iloc[i, 1] - 1
        rating = data_df.iloc[i, 2]
        if user not in user_list:
            user_list[user] = [[item, rating]]
        else:
            user_list[user].append([item, rating])

    train_file = open("Data/ml_train.txt", 'w', encoding='utf-8')
    test_file = open("Data/ml_test.txt", 'w', encoding='utf-8')
    for user, itemlist in user_list.items():
        for item in itemlist:
            r = random.random()
            if r < 0.2:
                test_file.write("%d,%d,%d\n" % (user, item[0], item[1]))
            else:
                train_file.write("%d,%d,%d\n" % (user, item[0], item[1]))

    train_file.close()
    test_file.close()

split_user()
