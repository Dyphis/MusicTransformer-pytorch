from model import MusicTransformer
import custom
from custom.metrics import *
from custom.criterion import SmoothCrossEntropyLoss, CustomSchedule
from custom.config import config
from data_single_file import Data, train_test_transposition

import utils
import datetime
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter


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
mt = MusicTransformer(
            embedding_dim=config.embedding_dim,
            vocab_size=config.vocab_size,
            num_layer=config.num_layers,
            max_seq=config.max_seq,
            dropout=config.dropout,
            debug=config.debug, loader_path=config.load_path
)
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

# Train Start
print(">> Train start...")
idx = 0
for e in range(config.epochs):
    print(">>> [Epoch was updated]")
    sw_start = time.time()
    for b in range(int(len(dataset.files) * 0.8) // (config.batch_size)):
        scheduler.optimizer.zero_grad()
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(config.batch_size, config.max_seq, batch_no = b)
            #batch_x, batch_y = train_test_transposition(batch_x, batch_y)
            batch_x = torch.from_numpy(batch_x).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
            batch_y = torch.from_numpy(batch_y).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
        except IndexError:
            continue

        start_time = time.time()
        mt.train()
        sample = mt.forward(batch_x)
        metrics = metric_set(sample, batch_y)
        loss = metrics['loss']
        loss.backward()
        scheduler.step()
        end_time = time.time()

        #if config.debug:
            #print("[Loss]: {}".format(loss))
        #if b == 0:
            #train_metrics = metrics
        #else:
            #train_metrics = utils.sum_dict(train_metrics, metrics)

        #record every 4 batch
        '''
        if b % 4 == 3:
            metrics = train_metrics
            train_summary_writer.add_scalar('loss', metrics['loss']/4, global_step=idx)
            train_summary_writer.add_scalar('accuracy', metrics['accuracy'][0]/4, global_step=idx)
            train_summary_writer.add_scalar('on_state_accuracy', metrics['accuracy'][1]/4, global_step=idx)
            train_summary_writer.add_scalar('off_state_accuracy', metrics['accuracy'][2]/4, global_step=idx)
            train_summary_writer.add_scalar('time_shift_accuracy', metrics['accuracy'][3]/4, global_step=idx)
            #train_summary_writer.add_scalar('velocity_accuracy', metrics['accuracy'][4], global_step=idx)
            train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=idx)
            #train_summary_writer.add_scalar('iter_p_sec', end_time-start_time, global_step=idx)
        '''
        train_summary_writer.add_scalar('loss', metrics['loss'], global_step=idx)
        train_summary_writer.add_scalar('accuracy', metrics['accuracy'][0], global_step=idx)
        train_summary_writer.add_scalar('on_state_accuracy', metrics['accuracy'][1], global_step=idx)
        train_summary_writer.add_scalar('off_state_accuracy', metrics['accuracy'][2], global_step=idx)
        train_summary_writer.add_scalar('time_shift_accuracy', metrics['accuracy'][3], global_step=idx)
        #train_summary_writer.add_scalar('velocity_accuracy', metrics['accuracy'][4], global_step=idx)
        train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=idx)
        #train_summary_writer.add_scalar('iter_p_sec', end_time-start_time, global_step=idx)
        torch.cuda.empty_cache()
        idx += 1

    #evaluate every epoch
    acc = 0
    on_acc = 0
    off_acc = 0
    time_acc = 0
    loss = 0
    for b in range(int(len(dataset.files) * 0.2) // (config.batch_size)):
        single_mt.eval()
        eval_x, eval_y = dataset.slide_seq2seq_batch(config.batch_size, config.max_seq, 'eval', b)
        eval_x = torch.from_numpy(eval_x).contiguous().to(config.device, dtype=torch.int)
        eval_y = torch.from_numpy(eval_y).contiguous().to(config.device, dtype=torch.int)

        #eval_preiction, weights = single_mt.forward(eval_x)
        eval_preiction = single_mt.forward(eval_x)

        metrics_temp = metric_set(eval_preiction, eval_y)
        
        loss += metrics_temp['loss']
        acc += metrics_temp['accuracy'][0]
        on_acc += metrics_temp['accuracy'][1]
        off_acc += metrics_temp['accuracy'][2]
        time_acc += metrics_temp['accuracy'][3]
        #if b == 0:
            #eval_metrics = metrics_temp
        #else:
            #eval_metrics = utils.sum_dict(eval_metrics, metrics)
            #train_summary_writer.add_histogram("target_analysis", batch_y, global_step=e)
            #train_summary_writer.add_histogram("source_analysis", batch_x, global_step=e)
            #for i, weight in enumerate(weights):
                #attn_log_name = "attn/layer-{}".format(i)
                #utils.attention_image_summary(
                    #attn_log_name, weight, step=idx, writer=eval_summary_writer)

    eval_summary_writer.add_scalar('loss', loss/(b+1), global_step=idx-1)
    #eval_summary_writer.add_scalar('accuracy', eval_metrics['accuracy'], global_step=idx)
    eval_summary_writer.add_scalar('accuracy', acc/(b+1), global_step=idx-1)
    eval_summary_writer.add_scalar('on_state_accuracy', on_acc/(b+1), global_step=idx-1)
    eval_summary_writer.add_scalar('off_state_accuracy', off_acc/(b+1), global_step=idx-1)
    eval_summary_writer.add_scalar('time_shift_accuracy', time_acc/(b+1), global_step=idx-1)
    #eval_summary_writer.add_scalar('velocity_accuracy', eval_metrics['accuracy'][4]/(b+1), global_step=idx-1)
    eval_summary_writer.add_histogram("logits_bucket", metrics_temp['bucket'], global_step=idx-1)
    train_summary_writer.add_histogram("target_analysis", eval_y, global_step=idx-1)
    train_summary_writer.add_histogram("source_analysis", eval_x, global_step=idx-1)

    print('Epoch/Batch: {}/{}'.format(e, idx))
    print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(metrics['loss'], metrics['accuracy']))
    print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(metrics_temp['loss'], metrics_temp['accuracy']))
    print('\n====================================================')
    torch.cuda.empty_cache()
    torch.save(single_mt.state_dict(), args.model_dir+'/train-{}.pth'.format(e))
        # switch output device to: gpu-1 ~ gpu-n
        #
        #if torch.cuda.device_count() > 1:
            #mt.output_device = idx % (torch.cuda.device_count() -1) + 1
    sw_end = time.time()
        #if config.debug:
    print('epoch time: {}'.format(sw_end - sw_start) )

torch.save(single_mt.state_dict(), args.model_dir+'/final.pth'.format(idx))
eval_summary_writer.close()
train_summary_writer.close()


