from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
from tensorboardX import SummaryWriter
import math
import datetime

from custom.metrics import *

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
'''
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
'''

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,x.size(1)]
        return self.dropout(x)


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
            num_encoder_layers: int,
            num_decoder_layers: int,
            emb_size: int,
            nhead: int,
            src_vocab_size: int,
            tgt_vocab_size: int,
            dim_feedforward: int = 512,
            dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        #self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
            src: Tensor,
            trg: Tensor,
            src_mask: Tensor,
            tgt_mask: Tensor,
            src_padding_mask: Tensor,
            tgt_padding_mask: Tensor,
            memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.tok_emb(src))
        tgt_emb = self.positional_encoding(self.tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
            self.tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            self.tok_emb(tgt)), memory,
            tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    #batch first: 0 -> 1
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


torch.manual_seed(0)

SRC_VOCAB_SIZE = 356
TGT_VOCAB_SIZE = 356
EMB_SIZE = 256
NHEAD = 4
FFN_HID_DIM = 512
BATCH_SIZE = 4
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
PAD_IDX = -100
MAX_LEN = 1024
NUM_EPOCHS = 100

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
            NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

# init metric set
metric_set = MetricsSet({
    'accuracy': CategoricalAccuracy(),
    #'loss': SmoothCrossEntropyLoss(0.1, SRC_VOCAB_SIZE, PAD_IDX),
    'bucket':  LogitsBucketting(SRC_VOCAB_SIZE)
})

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
#loss_fn = metric_set

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
"""
from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
            torch.tensor(token_ids),
            torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
            vocab_transform[ln], #Numericalization
            tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
"""

from torch.utils.data import DataLoader
from data_single_file import Data, train_test_transposition

# load data
dataset = Data('MusicTransformer-pytorch/dataset/processed/')
print(dataset)

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/'+'/'+current_time+'/train'
eval_log_dir = 'logs/'+'/'+current_time+'/eval'

train_summary_writer = SummaryWriter(train_log_dir)
eval_summary_writer = SummaryWriter(eval_log_dir)

def train_epoch(model, optimizer, batch_no):
    model.train()
    losses = 0
    #train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    #train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)


    for b in range(int(len(dataset.files)*0.8) // BATCH_SIZE):
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(BATCH_SIZE, MAX_LEN, batch_no = b)
            batch_x = torch.from_numpy(batch_x).contiguous().to(DEVICE, non_blocking=True, dtype=torch.long)
            batch_y = torch.from_numpy(batch_y).contiguous().to(DEVICE, non_blocking=True, dtype=torch.long)
        except IndexError:
            continue

        src = batch_x
        tgt = batch_y
        #src = src.to(DEVICE)
        #tgt = tgt.to(DEVICE)

        #tgt_input = tgt[:-1, :]
        tgt_input = src

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        #tgt_out = tgt[1:, :]
        tgt_out = tgt
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1).long())
        metrics = metric_set(logits, tgt_out)
        #loss = metrics['loss']
        loss.backward()

        optimizer.step()
        losses += loss.item()

        train_summary_writer.add_scalar('loss', loss, global_step=batch_no)
        train_summary_writer.add_scalar('accuracy', metrics['accuracy'][0], global_step=batch_no)
        train_summary_writer.add_scalar('on_state_accuracy', metrics['accuracy'][1], global_step=batch_no)
        train_summary_writer.add_scalar('off_state_accuracy', metrics['accuracy'][2], global_step=batch_no)
        train_summary_writer.add_scalar('time_shift_accuracy', metrics['accuracy'][3], global_step=batch_no)
        #train_summary_writer.add_scalar('velocity_accuracy', metrics['accuracy'][4], global_step=batch_no)
        #train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=batch_no)
        #train_summary_writer.add_scalar('iter_p_sec', end_time-start_time, global_step=batch_no)
        batch_no += 1
        print('batch {} was trained.'.format(b+1))
    return losses / (b+1), batch_no


def evaluate(model, batch_no):
    model.eval()
    losses = 0

    #val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    #val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)


    #for src, tgt in val_dataloader:
    for b in range(int(len(dataset.files)*0.2) // BATCH_SIZE):
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(BATCH_SIZE, MAX_LEN, batch_no = b)
            batch_x = torch.from_numpy(batch_x).contiguous().to(DEVICE, non_blocking=True, dtype=torch.long)
            batch_y = torch.from_numpy(batch_y).contiguous().to(DEVICE, non_blocking=True, dtype=torch.long)
        except IndexError:
            continue
        src = batch_x
        tgt = batch_y
        #src = src.to(DEVICE)
        #tgt = tgt.to(DEVICE)

        #tgt_input = tgt[:-1, :]
        tgt_input = src

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt
        #tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        metrics = metric_set(logits, tgt_out)

        eval_summary_writer.add_scalar('loss', loss, global_step=batch_no)
        eval_summary_writer.add_scalar('accuracy', metrics['accuracy'][0], global_step=batch_no)
        eval_summary_writer.add_scalar('on_state_accuracy', metrics['accuracy'][1], global_step=batch_no)
        eval_summary_writer.add_scalar('off_state_accuracy', metrics['accuracy'][2], global_step=batch_no)
        eval_summary_writer.add_scalar('time_shift_accuracy', metrics['accuracy'][3], global_step=batch_no)
    return losses / (b+1)


from timeit import default_timer as timer
batch_counter = 0
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss, batch_counter = train_epoch(transformer, optimizer, batch_counter)
    end_time = timer()
    val_loss = evaluate(transformer, batch_counter)
    #print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    print("Epoch: {}, Train loss: {:.3f}, Val loss: {:.3f}, Epoch time = {:.3f}s".format(epoch,train_loss,val_loss,(end_time - start_time)))
    torch.save(transformer.state_dict(), 'model/train-{}.pth'.format(epoch))
torch.save(transformer.state_dict(), 'model/final.pth')
