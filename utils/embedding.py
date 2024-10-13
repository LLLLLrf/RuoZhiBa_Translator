# Base on https://github.com/shibing624/text2vec
# # 中文句向量模型(CoSENT)，中文语义匹配任务推荐，支持fine-tune继续训练
# # 支持多语言的句向量模型（CoSENT），多语言（包括中英文）语义匹配任务推荐，支持fine-tune继续训练
# # 中文词向量模型(word2vec)，中文字面匹配任务和冷启动适用
# 暂时选用中文词向量模型(word2vec)，后续再改

from text2vec import Word2Vec


class Embedding:
    def __init__(self):
        self.model = Word2Vec("w2v-light-tencent-chinese")

    def encode(self, words: list):
        return self.model.encode(words)
