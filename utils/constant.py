"""
Define constants.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PER': 2, 'VEH': 3, 'LOC': 4, 'ORG': 5, 'GPE': 6, 'FAC': 7, 'WEA': 1}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'FAC': 2, 'PER': 3, 'VEH': 4, 'LOC': 5, 'WEA': 6, 'ORG': 7, 'GPE': 8}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'WEA': 3, 'VEH': 4, 'FAC': 5, 'GPE': 6, 'ORG': 7, 'PER': 8, 'LOC': 9}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PRP': 2, ':': 3, 'VBG': 4, 'JJ': 5, "''": 6, 'PDT': 7, 'RP': 8, '$': 9, '-LRB-': 10, '``': 11, 'WP$': 12, 'VB': 13, 'JJS': 14, 'NNPS': 15, '.': 16, 'RB': 17, 'VBZ': 18, 'WRB': 19, 'DT': 20, 'EX': 21, 'VBD': 22, 'RBS': 23, 'CC': 24, 'VBN': 25, 'NN': 26, 'MD': 27, 'FW': 28, 'NNS': 29, 'LS': 30, 'TO': 31, 'RBR': 32, 'VBP': 33, 'WDT': 34, 'JJR': 35, 'SYM': 36, 'WP': 37, 'CD': 38, 'PRP$': 39, 'NNP': 40, ',': 41, 'IN': 42, '-RRB-': 43, 'UH': 44, 'POS': 45}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'N': 2}

NEGATIVE_LABEL = 'NONE'

LABEL_TO_ID = {'NONE': 0, 'PART-WHOLE(e2,e1)': 1, 'ART(e1,e2)': 2, 'PART-WHOLE(e1,e2)': 3, 'PER-SOC(e1,e2)': 4, 'PER-SOC(e2,e1)': 5, 'ART(e2,e1)': 6, 'GEN-AFF(e2,e1)': 7, 'PHYS(e2,e1)': 8, 'PHYS(e1,e2)': 9, 'GEN-AFF(e1,e2)': 10, 'ORG-AFF(e2,e1)': 11, 'ORG-AFF(e1,e2)': 12}

INFINITY_NUMBER = 1e12

CHUNK_TO_ID = {'I-SBAR': 0, 'B-SBAR': 1, 'B-ADJP': 2, 'I-ADVP': 3, 'B-PP': 4, 'I-VP': 5, 'I-UCP': 18, 'I-CONJP': 7, 'I-PP': 8, 'O': 9, 'B-LST': 10, 'I-NP': 11, 'I-ADJP': 6, 'B-PRT': 13, 'B-INTJ': 14, 'B-ADVP': 15, 'B-NP': 16, 'I-INTJ': 17, 'B-VP': 12, 'B-UCP': 19, 'B-CONJP': 20, 'I-LST': 21}
