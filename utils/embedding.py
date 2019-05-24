import numpy as np

def load_bin_vec(fname, vocab={}, no_vocab=False):
   """
   Loads 300x1 word vecs from Google (Mikolov) word2vec
   """
   word_vecs = {}
   with open(fname, "rb") as f:
       header = f.readline()
       vocab_size, layer1_size = map(int, header.split())
       binary_len = np.dtype('float32').itemsize * layer1_size
       for line in range(vocab_size):
           print(line/float(vocab_size))
           word = []
           while True:
               ch = f.read(1)
               if ch == ' ':
                   word = ''.join(word)
                   break
               if ch != '\n':
                   word.append(ch)
           if word in vocab or no_vocab:
              word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
           else:
               f.read(binary_len)
   with open('dataset/word2vec/vectors.txt', 'w') as file:
        for k, v in word_vecs.items():
            file.write(k+" "+" ".join(map(str, v))+"\n")
   return word_vecs

def load_text_vec(fname, vocab, no_vocab=False):
   word_vecs = {}
   count = 0
   dim = 0
   with open(fname, 'r') as f:
       for line in f:
           count += 1
           line = line.strip()
           if count == 1:
               if len(line.split()) < 10:
                   dim = int(line.split()[1])
                   print('dim: ', dim)
                   continue
               else:
                   dim = len(line.split()) - 1
                   print('dim: ', dim)
           word = line.split()[0]
           emStr = line[(line.find(' ')+1):]
           if word in vocab or no_vocab:
               word_vecs[word] = np.fromstring(emStr, dtype='float32', sep=' ')
               if word_vecs[word].shape[0] != dim:
                   print('mismatch dimensions: ', dim, word_vecs[word].shape[0])
                   exit()
   print('loaded ', len(word_vecs), ' words in word embeddings')
   with open('dataset/concat/vectors.txt', 'w') as file:
        for k, v in word_vecs.items():
            file.write(k+" "+" ".join(map(str, v))+"\n")
   return dim, word_vecs

# load_text_vec('dataset/concat/gw-bolt.word2vecconcat.cbow1.sz300.w5.neg10.smpl1e-5.mincnt20.iter1', {}, True)
load_bin_vec('dataset/word2vec/GoogleNews-vectors-negative300.bin', {}, True)