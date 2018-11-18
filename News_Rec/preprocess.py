'''
对数据集的预处理，包括统计平均用户点击，新闻被点击次数 抽取用户-新闻对数据，和新闻id-内容数据 根据时间戳分割训练集和测试集
'''
# 强大的Python数据分析工具包
import pandas as pd
import time
# 正则化
import re
# gensim自然语言处理库 将文档根据TF-IDF, LDA, LSI模型转化成向量模式 此外，gensim还可以将单词转化为词向量
from gensim.models import ldamodel
# corpora是文档集的表现形式（二维矩阵 稀疏 有单词为1 无单词为0）（生成词典 可记录次数）
from gensim import corpora
# 导入jieba分词
import jieba
# analyse是关键词提取
from jieba import analyse
# 支持高级大量的维度数组与矩阵运算
import numpy as np
# 导入sklearn的特征提取包
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# 导入sklearn的计算余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
# 导入词性标注分词器
import jieba.posseg
import json

# 统计用户点击情况和新闻被点击次数
def statistic():
    data_df = pd.read_csv("Data/user_click_data.txt", sep='\t', header=-1) #读取scv数据 用\t分隔 到列表数据
    user_dict = dict() #创建空字典user_dict（存放用户读了的新闻个数）
    news_dict = dict() #创建空字典news_dict（新闻被用户点击的次数）
    for i in range(data_df.shape[0]):#循环取出列表每一行 data_df.shape[0]表示获得行数
        user_id = data_df.iloc[i, 0] #将第i行的第1列赋给user_id 用户编号（iloc表示通过行号索引行数据）
        news_id = data_df.iloc[i, 1] #将第i行的第2列赋给news_id 新闻编号
        
        if user_id not in user_dict: #如果取出来的这个user_id不在user_dict字典中
            user_dict[user_id] = 1 #将user_dict字典这个用户读的新闻数量赋成1（字典为用户ID：点击次数）
        else:
            user_dict[user_id] += 1 #否则在，将数量加1
        
        if news_id not in news_dict: #如果取出来的news_id不在news_dict字典中
            news_dict[news_id] = 1 #将字典的这篇文章的用户点击次数赋成1（字典中为新闻ID：被点击次数）
        else:
            news_dict[news_id] += 1 #否则在，将数量加1
    user_read_mean = sum(user_dict.values()) / len(user_dict) #用户读了的平均数量=所有用户读的新闻数量之和／用户个数
    news_read_mean = sum(news_dict.values()) / len(news_dict) #新闻平均点击次数=所有新闻被用户点击数之和／新闻个数
    min_userread = min(user_dict.values()) #存储最小的用户读新闻数量
    min_newsread = min(news_dict.values()) #存储最小的新闻被用户点击次数
    # 打印出来
    print("Mean user read times: %g" % user_read_mean)
    print("Mean news read times: %g" % news_read_mean)
    print("Min user read times: %d" % min_userread)
    print("Min news read times: %d" % min_newsread)

# 抽取新闻内容
def extract():
    data_df = pd.read_csv("Data/user_click_data.txt", sep='\t', header=-1) #读取scv数据 用\t分隔 到列表数据
    user_dict = dict() #创建空字典user_dict（存放用户读了的新闻个数）
    news_dict = dict() #创建空字典news_dict（新闻被用户点击的次数）
    uid = 0 #存放用户个数
    nid = 0 #存放新闻个数
    # 提取用户和新闻id,存储新闻内容
    news_info_file = open("Data/news_context.txt", 'w', encoding='utf-8') #将新闻内容写到news_context.txt
    for i in range(data_df.shape[0]):#循环取出列表每一行 data_df.shape[0]表示获得行数
        user_id = data_df.iloc[i, 0] #将第i行的第1列赋值给user_id 用户编号（iloc表示通过行号索引行数据）
        news_id = data_df.iloc[i, 1] #将第i行的第2列赋值给news_id 新闻编号
        
        if user_id not in user_dict: #如果取出来的user_id不在user_dict字典中
            user_dict[user_id] = uid #在user_dict字典中添加（user_id：uid（第几个用户））
            uid += 1 #将用户个数加1
    
        if news_id not in news_dict: #如果取出来的news_id不在news_dict字典中
            news_dict[news_id] = nid #在news_dict字典中添加（news_id：nid（第几篇新闻））
            context = [str(nid), str(data_df.iloc[i, 3]), str(data_df.iloc[i, 4]), str(data_df.iloc[i, 5])] #内容为新闻第几个 标题 内容 发布时间
            news_info_file.write('\t'.join(context) + '\n') #在news_info_file中增加这一行的新闻内容
            nid += 1 #将新闻个数加1
    news_info_file.close()

    # 存储用户id字典和新闻id字典
    user_list = sorted(user_dict.items(), key=lambda e: e[1], reverse=False) #items方法将字典的元素转化为元组 选取元组的第二个元素比较参数（key仍在第一位）
    news_list = sorted(news_dict.items(), key=lambda e: e[1], reverse=False)
    
    #将用户id存入user_id.txt中
    user_file = open("Data/user_id.txt", 'w', encoding='utf-8')
    for user in user_list:
        user_file.write(str(user[1]) + '\t' + str(user[0]) + '\n')
    user_file.close()
    #将新闻id存入news_id.txt中
    news_file = open("Data/news_id.txt", 'w', encoding='utf-8')
    for news in news_list:
        news_file.write(str(news[1]) + '\t' + str(news[0]) + '\n')
    news_file.close()

    # 存储用户号-新闻号-时间数据
    user_news_df = data_df.loc[:, 0:2] #loc为通过行标签索引行数据，将原始数据data_df按行取出所有行的用户id，新闻id，访问页面时间
    for i in range(data_df.shape[0]): #循环取出列表每一行 data_df.shape[0]表示获得行数
        user_news_df.iloc[i, 0] = user_dict[user_news_df.iloc[i, 0]] #用户id在用户id字典中的用户号赋给用户id
        user_news_df.iloc[i, 1] = news_dict[user_news_df.iloc[i, 1]] #新闻id在新闻id字典中的新闻号赋给新闻id
    user_news_df.to_csv("Data/user_news_id.csv", sep='\t', header=False, index=False) #将用户号 新闻号 时间存储下来

# 时间戳转日期
def timestamp_datatime(value):
    format = '%Y-%m-%d %H:%M'
    #format = '%Y-%m-%d %H:%M:%S'
    #value 为时间戳值,如:1460073600.0
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

# 按时间分割测试集和训练集
def split_data():
    data_df = pd.read_csv("Data/user_news_id.csv", header=-1, sep='\t') #将用户号-新闻号-读新闻时间 文件打开
    train_file = open("Data/train_data.txt", 'w', encoding='utf-8')
    test_file = open("Data/test_data.txt", 'w', encoding='utf-8')
    for i in range(data_df.shape[0]): #按照行循环
        read_time = data_df.iloc[i, 2] #取每行读新闻时间那一列
        date = timestamp_datatime(read_time) #将时间戳转为日期
        day = date.split(' ')[0].split('-')[2] #按照空格分隔的第一个部分的第三个数取出来赋值给day
        uid = str(data_df.iloc[i, 0]) #将用户号取出来
        nid = str(data_df.iloc[i, 1]) #将新闻号取出来
        if int(day) < 20: #如果day小于20 就将uid nid 时间写入训练集合
            train_file.write(uid + '\t' + nid + '\t' + date + '\n')
        else: #大于等于20 写入uid nid 时间写入测试集合
            test_file.write(uid + '\t' + nid + '\t' + date + '\n')
    train_file.close()
    test_file.close()


def get_delwords(path):
    '''获取需要剔除的词表'''
    with open(path, 'r', encoding='utf-8') as file:
        return set([line.strip('\n') for line in file])

# 提取新闻内容中的主题 对新闻内容进行ida主题匹配
def extract_topic():
    ctx_data = pd.read_csv("Data/news_context.txt", sep='\t', header=-1) #打开新闻内容文件
    stop_set = get_delwords("Data/stopwords.dat") #得到停用词表
    symbol_set = get_delwords("Data/symbol_list.txt") #得到符号表
    words_list = [] #定义一个空列表
    for i in range(len(ctx_data)): #循环新闻内容文档的每一个文档
        news_ctx = str(ctx_data.iloc[i, 1]) + ' ' + str(ctx_data.iloc[i, 2]) #取出新闻的名字和内容
        news_words = list(jieba.posseg.cut(news_ctx)) #对他们进行分词并标注词性
        words = list() #定一个空列表
        for wrd in news_words: #循环每个分好的词语
            term = wrd.word
            flag = wrd.flag
            if term not in stop_set and term not in symbol_set: #如果词语不在停用词表中 并且不是符号
                isnum = re.match(r'([a-z0-9A-Z%\._]+)', term) #将term和模式串匹配（去掉数字字母）
                if isnum is None: #如果没有匹配
                    if 'n' in flag or 'v' in flag: #如果类型是n或者v
                        words.append(term) #将词加入到列表words中
        words_list.append(words) #一篇文档结束 将文档的words加入到整个的列表中

    # 制作词典
    word_dict = corpora.Dictionary(words_list) #文档集的表现形式（二维矩阵 稀疏 有单词为1 无单词为0）（生成词典，可记录次数）
    word_dict.save_as_text("Data/vocab.txt") #将词典存放到vocab.txt中
    # 得到一个主题的序号和内容
    def get_dotcontent(content):
        str_list = []
        dotindex = -1
        while dotindex + 1 < len(content):
            dotindex = dotindex + content[dotindex + 1:].index('"') + 1
            s = dotindex
            dotindex = content[dotindex + 1:].index('"') + dotindex + 1
            e = dotindex
            str_list.append(content[s + 1:e])

        return str_list

    # 语料列表
    corpus_list = [word_dict.doc2bow(text) for text in words_list] #将每篇新闻中包含词语转化成词典word_dict 进行词袋转化doc2bow
    # 构建lda模型 词袋为corpus_list 词典为word_dict映射为字符串 生成主题的个数50 模型遍历语料库的次数为1000次
    lda = ldamodel.LdaModel(corpus=corpus_list, id2word=word_dict, num_topics=50, alpha='auto', iterations=1000)
    with open("Data/topic.txt", 'w', encoding='utf-8') as file: #主题写topic.txt（13说 警察 货币 钱 老百姓 时 近东 买 想 流民）
        for topic in lda.show_topics(num_topics=50):
            file.write(str(topic[0]) + '\t' + ' '.join(get_dotcontent(topic[1])) + '\n')# 将序号和主题链接一起
    with open("Data/news_topic.txt", 'w', encoding='utf-8') as file: #
        topic_dict = dict() #新建一个字典
        for n in range(len(corpus_list)): #循环词袋（文章个数）
            topic = lda.get_document_topics(corpus_list[n]) #每篇新闻与50个主题的相似度
            topic_dict[n] = topic #将字典中加入上面所求
    file.write(json.dumps(topic_dict)) #在file中保存字典
    lda.save("trained_model/lda_model") #将ida主题模型保存下来

# 将新闻内容生成tfidf矩阵, 直接计算并保存相似度矩阵
def gen_tfidf():
    ctx_data = pd.read_csv("Data/news_context.txt", sep='\t', header=-1) #打开新闻内容的文档
    stop_set = get_delwords("Data/stopwords.dat") #打开停用词表
    symbol_set = get_delwords("Data/symbol_list.txt") #打开符号表
    corpus = [] #新建一个列表
    for i in range(len(ctx_data)): #循环每一篇文档
        print(i) #打印是第几篇文档
        news_ctx = str(ctx_data.iloc[i, 1]) + ' ' + str(ctx_data.iloc[i, 2]) #将新闻标题和内容提取出来
        news_words = list(jieba.cut(news_ctx)) #用结巴进行分词
        words = list() #新建一个列表
        for term in news_words: #循环每一个词
            if term not in stop_set and term not in symbol_set: #如果不是停用词和符号
                isnum = re.match(r'([a-z0-9A-Z%\._]+)', term) #去掉字母和数字
                if isnum is None: #如果不能匹配
                    words.append(term) #将词语放到words列表中
        corpus.append(' '.join(words)) #将每篇文档的列表加入到总的列表中corpus
    CV = CountVectorizer()
    TFIDF = TfidfTransformer()
    tf_mat = CV.fit_transform(corpus) #将切好的文档中的词进行文档词频矩阵转化（A：1 3 2 1 8）
    tfidf_mat = TFIDF.fit_transform(tf_mat) #将文档词频转化成tfidf稀疏矩阵（矩阵中的值为每一个词在这篇文档中的tfidf值）（A：0 0 0.23 0.34 0.28 0.1 0.6）（所有的词都有 维度相同）
    print(np.shape(tfidf_mat))
    # np.savetxt("Data/tfidf.mat", tfidf_mat)
    sim_mat = cosine_similarity(tfidf_mat) #计算两篇文档之间的相似度矩阵
    print(np.shape(sim_mat))
    np.savetxt("Data/news_sim.mat", sim_mat) #将文档的余弦相似度矩阵保存在news_sim.mat中
    print("Sim mat saved")

# 找冷启动用户
def cold_user():
    data_df = pd.read_csv("Data/train_data.txt", sep='\t', header=-1) #打开训练数据
    test_df = pd.read_csv("Data/test_data.txt", sep='\t', header=-1) #打开测试数据
    user_set = set() #定义一个空集合
    test_set = set()
    for i in range(len(data_df)): #对于训练数据中每条记录
        user_set.add(data_df.iloc[i, 0]) #取用户ID那一列加入到集合中
    for i in range(len(test_df)): #对于测试数据中每条记录
        test_set.add(test_df.iloc[i, 0]) #取用户ID那一列加入到集合中
    cold_sum = 0 #计数
    for user in test_set: #对于测试数据中的每个用户
        if user not in user_set: #如果不在训练数据中
            cold_sum += 1 #数量加1
    print("User Cold Sum: %d" % cold_sum)

# 每篇文章都用词语序号表示
def gen_voc_content():
    vocab_file = open("Data/vocab.txt", 'r', encoding='utf-8') #打开词频统计表
    vocab = dict() #新建一个字典
    k = 1
    for line in vocab_file: #对于表中的每一行
        seq = line.split('\t') #分割每行并存入seq中
        if len(seq) > 1: #如果seq的长度大于1
            word = seq[1] #取出词那一列
            if word not in vocab.keys(): #如果词不在vocab字典中
                vocab[word] = k #将词加入字典 并将序号置成k（word：k）
                k += 1 #计数加1
    vocab_file.close()
    content_file = open("Data/news_context.txt", 'r', encoding='utf-8') #打开新闻内容的文档
    id_content_file = open("Data/news_context_id.txt", 'w', encoding='utf-8') #将新闻中的词用在词标中的序号表示
    maxlen = 0
    for line in content_file: #循环每篇新闻
        ctx_words = [] #定义一个空列表
        seq = line.split('\t') #将每篇文档分割开
        news_id = seq[0] #选择文档的第一列 表示文档号
        news_ctx = seq[1] + ' ' + seq[2] #将标题和内容取出
        ctx_cut = jieba.cut(news_ctx) #分词
        for word in ' '.join(ctx_cut).split(' '): #每篇文档分好词后循环每一个词
            if word in vocab.keys(): #如果词在vocab字典中
                ctx_words.append(str(vocab[word])) #将字典中词语的序号加入列表
        maxlen = max(len(ctx_words), maxlen) #maxlen中存放词语个数的最大数
        if len(ctx_words) == 0: #如果长度为0 就将0加入到列表
            ctx_words.append('0')
                id_content_file.write(str(news_id) + '\t' + ' '.join(ctx_words) + '\n') #每篇文章对应的序号和词语序号写入
    id_content_file.close()
    content_file.close()
    print("Maxlen: " + str(maxlen))

extract_topic() #提取新闻内容中的主题 对新闻内容进行ida主题匹配
