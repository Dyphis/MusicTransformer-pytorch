from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
from tensorboardX import SummaryWriter
import math
import datetime

from midi_processor.processor import decode_midi, encode_midi
from seq2seq import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, prior):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    #ys = torch.ones(1, 1).fill_(prior).type(torch.long).to(DEVICE)
    ys = prior
    
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys[:,i:].size(1))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys[:,i:], memory, tgt_mask)
        #out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        #print(next_word)
        next_word = next_word.item()
        print('output',next_word)
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        #if next_word == EOS_IDX:
            #break
    return ys

# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src: torch.Tensor):
    model.eval()
    #src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 400, prior=src)
    return tgt_tokens.contiguous().tolist()

SRC_VOCAB_SIZE = 356
TGT_VOCAB_SIZE = 356
EMB_SIZE = 256
NHEAD = 4
FFN_HID_DIM = 512
BATCH_SIZE = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
PAD_IDX = -100
MAX_LEN = 1024

mt = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
            NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
mt.load_state_dict(torch.load('final.pth'))
mt.to(DEVICE)

inputs = np.array([encode_midi('MusicTransformer-pytorch/dataset/midi/MarioTheme.mid')[:1024]])
inputs = torch.from_numpy(inputs)
result = translate(mt, inputs)

for i in result:
    print(i)

decode_midi(result, file_path='MusicTransformer-pytorch/midi_processor/bin/generated.mid')