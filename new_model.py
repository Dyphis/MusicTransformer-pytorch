import torch
import torch.nn.functional as F

from progress.bar import Bar


#define the model
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
        self.mt = torch.nn.Transformer(
            d_model=512, nhead=8, num_encoder_layers=6,
            num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
            activation='relu', custom_encoder=None, custom_decoder=None,
            layer_norm_eps=1e-05, batch_first=False, device=None, dtype=None)
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.fc = torch.nn.Linear(self.embedding_dim, self.vocab_size)

        def forward(self, src, tgt, length=None, writer=None):
            src_mask = torch.nn.transformer.generate_square_subsequent_mask(self.embedding_dim)


        def forward(self, x, length=None, writer=None):
            if self.training or not self.infer:
                #_, _, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x, config.pad_token)
                decoder, w = self.Decoder(x, mask=look_ahead_mask)
                fc = self.fc(decoder)
                return fc.contiguous() if self.training else (
                fc.contiguous(), [weight.contiguous() for weight in w])
            else:
                return self.generate(x, length, None).contiguous().tolist()

        def generate(self,
                     prior: torch.Tensor,
                     length=2048,
                     tf_board_writer: SummaryWriter = None):




        def test(self):
            self.eval()
            self.infer = True
