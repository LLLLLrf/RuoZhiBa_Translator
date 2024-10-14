import sentencepiece as spm

text_path = "data/sentence.txt"
output_folder = "weights/"

spm.SentencePieceTrainer.train(
    '--input=' + text_path + ' --model_prefix=' + output_folder + 'sentencepiece_bpe --vocab_size=32000 --character_coverage=0.9995 --model_type=bpe')

# spm.SentencePieceTrainer.train(
#     '--input=' + text_path + ' --model_prefix=' + output_folder + 'sentencepiece_unigram --vocab_size=16000 --character_coverage=0.9995 --model_type=unigram')

# spm.SentencePieceTrainer.train(
#     '--input=' + text_path + ' --model_prefix=' + output_folder + 'sentencepiece_char --vocab_size=16000 --character_coverage=0.9995 --model_type=char')
