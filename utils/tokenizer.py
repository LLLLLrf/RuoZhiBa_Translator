import sentencepiece as spm


class Tokenizer(spm.SentencePieceProcessor):
    def __init__(self, mode="bpe", model_path="weights/sentencepiece_"):
        super().__init__()
        self.Load(model_path + mode + ".model")

    def encode(self, sentence):
        return self.encode_as_ids(sentence)

    def decode(self, arr):
        if len(arr) == 0:
            return ""
        if type(arr[0]) == str:
            return self.decode_pieces(arr)
        return self.decode_ids(arr)


if __name__ == "__main__":
    tokenizer = Tokenizer(mode="bpe")

    # # encode: text => id
    print(tokenizer.encode('你好是一个汉语词语'))

    # # decode: id => text
    print(tokenizer.decode([2116, 26848, 93, 2888, 1665]))

    # # # encode: text => id
    # print(tokenizer.encode_as_pieces('你好是一个汉语词语'))
    # print(tokenizer.encode_as_ids('你好是一个汉语词语'))

    # # # decode: id => text
    # print(tokenizer.decode_pieces(
    #     ['你', '好', '是', '一', '个', '汉', '语', '词', '语']))
    # print(tokenizer.decode_ids([697, 13337, 56, 2912, 659]))
