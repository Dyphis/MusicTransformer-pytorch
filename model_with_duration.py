from custom.layers import *
from custom.criterion import *
from custom.layers import Encoder
from custom.config import config
from midi_processor.processor import *

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
        self.fc_combined1 = torch.nn.Linear(self.embedding_dim + 128, self.embedding_dim)
        self.fc_combined2 = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x, length=None, writer=None):
        if self.training or not self.infer:
            #print(x.shape)
            duration_info = self.duration_record(x)
            _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x, config.pad_token)
            decoder, w = self.Decoder(x, mask=look_ahead_mask)
            #print('decoder shape:', decoder.shape)
            #fc = decoder
            #print(duration_info.shape)
            concat_out = torch.cat((decoder, duration_info.to(decoder.device)), 2)
            fc = self.fc_combined1(concat_out)
            fc = self.fc_combined2(fc)
            fc = self.fc(fc)
            #print('fc shape:', fc.shape)
            return fc.contiguous() if self.training else (fc.contiguous(), [weight.contiguous() for weight in w])
        else:
            return self.generate(x, length, None).contiguous().tolist()

    def duration_record(self, idx_array):
        final_list = []
        for batch in range(idx_array.shape[0]):
            duration_list = []
            note_on_dict = [0 for _ in range(128)]
            empty_duration_list = [0 for _ in range(128)]
            duration_list.append(empty_duration_list)
            for idx in idx_array[batch].tolist():
                if idx in range(0, 128):
                    note_on_dict[idx] = 1 #mark as on
                    temp = duration_list[-1][:]
                    duration_list.append(temp)
                elif idx in range(128, 256):
                    note_on_dict[idx - 128] = 0 #mark as off
                    temp = empty_duration_list[:]
                    temp[idx - 128] = 0 #reset the duration
                    duration_list.append(temp)
                elif idx in range(256, 356):
                    if duration_list == []:
                        duration_list.append(empty_duration_list) #initialization
                        continue
                    for i in range(128):
                        temp = duration_list[-1][:]
                        temp[i] += note_on_dict[i] * ((idx - 256 + 1) /100)
                    duration_list.append(temp)
                else:
                    temp = duration_list[-1][:]
                    duration_list.append(temp)
            duration_list = duration_list[1:]
            print('duration list shape:', len(duration_list))
            final_list.append(duration_list)
        #print('final list shape:', len(final_list))
        return torch.Tensor(final_list)
    def generate(self,
                 prior: torch.Tensor,
                 length=2048,
                 tf_board_writer: SummaryWriter = None):
        decode_array = prior
        result_array = prior
        print(config)
        print(length)
        for i in Bar('generating').iter(range(length)):
            if decode_array.size(1) >= config.threshold_len:
                decode_array = decode_array[:, 1:]
            _, _, look_ahead_mask = \
                utils.get_masked_with_pad_tensor(decode_array.size(1), decode_array, decode_array, pad_token=config.pad_token)

            # result, _ = self.forward(decode_array, lookup_mask=look_ahead_mask)
            # result, _ = decode_fn(decode_array, look_ahead_mask)
            duration_info = self.duration_record(decode_array)
            result, _ = self.Decoder(decode_array, None)
            concat_out = torch.cat((result, duration_info.to(result.device)), 2)
            fc = self.fc_combined1(concat_out)
            fc = self.fc_combined2(fc)
            result = self.fc(fc)
            result = result.softmax(-1)

            if tf_board_writer:
                tf_board_writer.add_image("logits", result, global_step=i)

            u = 0
            if u > 1:
                result = result[:, -1].argmax(-1).to(decode_array.dtype)
                decode_array = torch.cat((decode_array, result.unsqueeze(-1)), -1)
            else:
                pdf = dist.OneHotCategorical(probs=result[:, -1])
                result = pdf.sample().argmax(-1).unsqueeze(-1)
                # result = torch.transpose(result, 1, 0).to(torch.int32)
                decode_array = torch.cat((decode_array, result), dim=-1)
                result_array = torch.cat((result_array, result), dim=-1)
            del look_ahead_mask
        result_array = result_array[0]
        return result_array

    def test(self):
        self.eval()
        self.infer = True
