from model import MusicTransformer
import custom
from custom.metrics import *
from custom.criterion import SmoothCrossEntropyLoss, CustomSchedule
from custom.config import config
from custom.layers import *
from data import Data

import utils
import datetime
import time
import math

import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tensorboardX import SummaryWriter
from progress.bar import Bar



# set config
parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')


# load data
dataset = Data(config.pickle_dir)
print(dataset)

# load model
learning_rate = config.l_r

# define model



class New_MusicTransformer(torch.nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        #self.pos_encoder = PositionalEncoding(d_model, dropout, max_len = config.max_seq)
        self.pos_encoder = DynamicPositionEmbedding(d_model, config.max_seq)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
        
    def test(self):
        self.eval()
        self.infer = True

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


ntokens = config.vocab_size  # size of vocabulary config.vocab_size
emsize = config.embedding_dim  # embedding dimension
d_hid = config.embedding_dim  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = config.num_layers  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = config.embedding_dim // 64  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability

mt = New_MusicTransformer(ntokens, emsize, nhead, d_hid, nlayers, dropout)

mt.to(config.device)
opt = optim.Adam(mt.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(config.embedding_dim, optimizer=opt)

# multi-GPU set
if torch.cuda.device_count() > 1:
    single_mt = mt
    mt = torch.nn.DataParallel(mt, output_device=torch.cuda.device_count()-1)
else:
    single_mt = mt

# init metric set
metric_set = MetricsSet({
    'accuracy': CategoricalAccuracy(),
    'loss': SmoothCrossEntropyLoss(config.label_smooth, config.vocab_size, config.pad_token),
    'bucket':  LogitsBucketting(config.vocab_size)
})

print(mt)
print('| Summary - Device Info : {}'.format(torch.cuda.device))

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/'+config.experiment+'/'+current_time+'/train'
eval_log_dir = 'logs/'+config.experiment+'/'+current_time+'/eval'

train_summary_writer = SummaryWriter(train_log_dir)
eval_summary_writer = SummaryWriter(eval_log_dir)

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

src_mask = generate_square_subsequent_mask(config.max_seq)
# Train Start
print(">> Train start...")
idx = 0
for e in range(config.epochs):
    print(">>> [Epoch was updated]")
    sw_start = time.time()
    for b in range(len(dataset.files) // config.batch_size):
        scheduler.optimizer.zero_grad()
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(config.batch_size, config.max_seq)
            batch_x = torch.from_numpy(batch_x).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
            batch_y = torch.from_numpy(batch_y).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
        except IndexError:
            continue

        start_time = time.time()
        mt.train()
        sample = mt.forward(batch_x, src_mask)
        metrics = metric_set(sample, batch_y)
        loss = metrics['loss']
        loss.backward()
        scheduler.step()
        end_time = time.time()

        if config.debug:
            print("[Loss]: {}".format(loss))

        train_summary_writer.add_scalar('loss', metrics['loss'], global_step=idx)
        train_summary_writer.add_scalar('accuracy', metrics['accuracy'], global_step=idx)
        train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=idx)
        train_summary_writer.add_scalar('iter_p_sec', end_time-start_time, global_step=idx)

        # result_metrics = metric_set(sample, batch_y)
        if b % 100 == 0:
            single_mt.eval()
            eval_x, eval_y = dataset.slide_seq2seq_batch(2, config.max_seq, 'eval')
            eval_x = torch.from_numpy(eval_x).contiguous().to(config.device, dtype=torch.int)
            eval_y = torch.from_numpy(eval_y).contiguous().to(config.device, dtype=torch.int)

            #eval_preiction, weights = single_mt.forward(eval_x, src_mask)
            eval_preiction = single_mt.forward(eval_x, src_mask)

            eval_metrics = metric_set(eval_preiction, eval_y)
            torch.save(single_mt.state_dict(), args.model_dir+'/train-{}.pth'.format(e))
            if b == 0:
                train_summary_writer.add_histogram("target_analysis", batch_y, global_step=e)
                train_summary_writer.add_histogram("source_analysis", batch_x, global_step=e)
                #for i, weight in enumerate(weights):
                    #attn_log_name = "attn/layer-{}".format(i)
                    #utils.attention_image_summary(
                        #attn_log_name, weight, step=idx, writer=eval_summary_writer)

            eval_summary_writer.add_scalar('loss', eval_metrics['loss'], global_step=idx)
            eval_summary_writer.add_scalar('accuracy', eval_metrics['accuracy'], global_step=idx)
            eval_summary_writer.add_histogram("logits_bucket", eval_metrics['bucket'], global_step=idx)

            print('\n====================================================')
            print('Epoch/Batch: {}/{}'.format(e, b))
            print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(metrics['loss'], metrics['accuracy']))
            print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_metrics['loss'], eval_metrics['accuracy']))
        torch.cuda.empty_cache()
        idx += 1

        # switch output device to: gpu-1 ~ gpu-n

        #if torch.cuda.device_count() > 1:
            #mt.output_device = idx % (torch.cuda.device_count() -1) + 1
    sw_end = time.time()
        #if config.debug:
    print('epoch time: {}'.format(sw_end - sw_start) )

torch.save(single_mt.state_dict(), args.model_dir+'/final.pth'.format(idx))
eval_summary_writer.close()
train_summary_writer.close()


