# Base on https://github.com/shibing624/text2vec
# # 中文句向量模型(CoSENT)，中文语义匹配任务推荐，支持fine-tune继续训练
# # 支持多语言的句向量模型（CoSENT），多语言（包括中英文）语义匹配任务推荐，支持fine-tune继续训练
# # 中文词向量模型(word2vec)，中文字面匹配任务和冷启动适用
# 暂时选用中文词向量模型(word2vec)，后续再改

from text2vec import Word2Vec
from text2vec import SentenceModel


class Embedding:
    def __init__(self):
        # self.model = Word2Vec("w2v-light-tencent-chinese")
        self.model = SentenceModel("shibing624/text2vec-base-chinese")

    def encode(self, words: list):
        return self.model.encode(words)
<<<<<<< HEAD
        
import numpy
if __name__ == "__main__":
    embedding = Embedding()
    lst=[]
    test_list=[["你好我","是一个" ,"人"],["你好","我是", "一个人"], ['这是一', '段', '测试', '文本', ',', '用于', '查看', '分词', '效果']]
    for i in test_list:
        lst.extend(embedding.encode(i))
    zero=[]
    for i, vec in enumerate(lst):
        i = numpy.array(i)
        if numpy.unique(vec).shape[0] == 1:
            zero.append(i)
    test_list = [j for i in test_list for j in i]
    for i in zero:
        print(test_list[i])
        
=======

    def decode(self, vectors):
        return self.model.decode(vectors)


if __name__ == "__main__":
    # model = Embedding()
    # a = model.encode(['你好', '是', '一个', '汉语', '词语'])
    # print(a)
    # print(model.decode(a))

    # 中文句向量模型(CoSENT)，中文语义匹配任务推荐，支持fine-tune继续训练
    t2v_model = SentenceModel("shibing624/text2vec-base-chinese")
    b = t2v_model.encode(['你好', '是', '一个', '汉语', '词语'])
    # b = t2v_model.encode(['你好是一个汉语词语'])
    print(b)
>>>>>>> c2ffc574fffc6b31f4c4517fdd8c3617c20bf466
