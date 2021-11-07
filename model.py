from custom.layers import *
from custom.criterion import *
from custom.layers import Encoder
from custom.config import config
from Beam import init_vars, k_best_outputs
import sys
import torch
import torch.distributions as dist
import random
import utils

import torch
from tensorboardX import SummaryWriter
from progress.bar import Bar


class MusicTransformer(torch.nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, debug=False, loader_path=None, dist=False, writer=None):
        super().__init__()
        self.infer = False
        if loader_path is not None:
            self.load_config_file(loader_path)
        else:
            self._debug = debug
            self.max_seq = max_seq
            self.num_layer = num_layer
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size
            self.dist = dist

        self.writer = writer
        self.Decoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.fc = torch.nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x, length=None, writer=None):
        if self.training or not self.infer:
            _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x, config.pad_token)
            #decoder, w = self.Decoder(x, mask=look_ahead_mask)
            decoder = self.Decoder(x, mask=look_ahead_mask)
            fc = self.fc(decoder)
            return fc.contiguous()# if self.training else (fc.contiguous(), [weight.contiguous() for weight in w])
        else:
            return self.generate(x, length, writer).contiguous().tolist()

    def generate(self,
                 prior: torch.Tensor,
                 length=2048,
                 ground_truth: torch.Tensor = None):
        #ground_truth = torch.Tensor(ground_truth)
        result_array = prior
        decode_array = prior
        print(config)
        print(length)
        for i in Bar('generating').iter(range(length)):
            if decode_array.size(1) > config.threshold_len:
                decode_array = decode_array[:, 1:]
            _, _, look_ahead_mask = \
                utils.get_masked_with_pad_tensor(decode_array.size(1), decode_array, decode_array, pad_token=config.pad_token)

            # result, _ = self.forward(decode_array, lookup_mask=look_ahead_mask)
            # result, _ = decode_fn(decode_array, look_ahead_mask)
            #result, _ = self.Decoder(decode_array, None)
            result = self.Decoder(decode_array, None)
            result = self.fc(result)
            result = result.softmax(-1)

            #if tf_board_writer:
                #tf_board_writer.add_image("logits", result, global_step=i)

            u = 2
            if u == 1:
                #greedy decoding method
                result = result[:, -1].argmax(-1).to(decode_array.dtype)
                #ground truth
                
                decode_array = torch.cat((decode_array, result.unsqueeze(-1)), -1)
                result_array = torch.cat((result_array, result.unsqueeze(-1)), -1)
                #decode_array = ground_truth[:, max(prior.shape[1]+i+1 - config.threshold_len, 0):prior.shape[1]+i+1]
            elif u == 2:
                #beam search decoding method
                k = 3
                #print(i)
                if i == 0:
                    result_array, log_scores = init_vars(decode_array, result, prior.shape[1] + length, k)
                    decode_array = result_array[:,max(prior.shape[1]+i+1 - config.threshold_len, 0):prior.shape[1]+1]
                    #print('initial array:',decode_array)
                    #print('input array:', result_array.shape)
                else:
                    result_array, log_scores = k_best_outputs(result_array, result, log_scores, prior.shape[1]+i, k)
                    decode_array = result_array[:, max(prior.shape[1]+i+1 - config.threshold_len, 0):prior.shape[1]+i+1]
                    #print('result array:', result_array)
                    #print('decode array:',decode_array)
            else: 
                #pdf decoding method
                pdf = dist.OneHotCategorical(probs=result[:, -1])
                result = pdf.sample().argmax(-1).unsqueeze(-1)
                # result = torch.transpose(result, 1, 0).to(torch.int32)
                decode_array = torch.cat((decode_array, result), dim=-1)
                result_array = torch.cat((result_array, result), dim=-1)
            del look_ahead_mask
        print(result_array)
        result_array = result_array[0]
        return result_array.long()

    def test(self):
        self.eval()
        self.infer = True
