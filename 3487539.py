#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U paddlenlp')


# In[2]:


get_ipython().system('tar -xvf data/data96333/商品评论情感预测.gz')


# In[4]:


from paddlenlp.datasets import load_dataset
from paddle.io import Dataset, Subset
from paddlenlp.datasets import MapDataset
import re

# 数据ID,用户ID,商品ID,评论时间戳,评论标题,评论内容,评分
def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as in_f:
        next(in_f)
        for line in in_f:
            line = line.strip('\n')
            split_array = [i.start() for i in re.finditer(',', line)]
            id = line[:split_array[0]]
            comment_title = line[split_array[3] + 1:split_array[4]]
            comment = line[split_array[4] + 2:split_array[-2]]
            label = line[split_array[-1] + 1:]
            yield {'text': comment_title  +' '+ comment, 'label': str(int(label.split('.')[0])-1), 'qid': id}

# 数据ID,用户ID,商品ID,评论时间戳,评论标题,评论内容,评分
def read_test(data_path):
    with open(data_path, 'r', encoding='utf-8') as in_f:
        next(in_f)
        for line in in_f:
            line = line.strip('\n')
            split_array = [i.start() for i in re.finditer(',', line)]
            id = line[:split_array[0]]
            id=id.split('_')[-1]
            comment_title = line[split_array[3] + 1:split_array[4]]
            comment = line[split_array[4] + 2:split_array[-2]]
            label= '1'
            yield {'text': comment_title  +' '+ comment, 'label': label, 'qid': id}

            


# In[5]:



# data_path为read()方法的参数
dataset_ds = load_dataset(read, data_path='训练集.csv',lazy=False)
# 在这进行划分
train_ds = Subset(dataset=dataset_ds, indices=[i for i in range(len(dataset_ds)) if i % 10 != 1])
dev_ds = Subset(dataset=dataset_ds, indices=[i for i in range(len(dataset_ds)) if i % 10 == 1])

test_ds =  load_dataset(read_test, data_path='测试集.csv',lazy=False)
for i in range(5):
    print(test_ds[i])


# In[6]:



# 在转换为MapDataset类型
train_ds = MapDataset(train_ds)
dev_ds = MapDataset(dev_ds)
test_ds = MapDataset(test_ds)
print(len(train_ds))
print(len(dev_ds))
print(len(test_ds))


# In[7]:


from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
model = SkepForSequenceClassification.from_pretrained(
    'skep_ernie_1.0_large_ch', num_classes=5)
tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')


# **数据 NLP 特征处理**

# In[8]:


get_ipython().system('pip install utils')


# In[9]:


import os
import paddle
import paddle.nn.functional as F 
from paddlenlp.data import Stack, Tuple, Pad
from utils import create_dataloader

def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    encoded_inputs = tokenizer(text=example['text'], max_seq_length=max_seq_length)
    input_ids = encoded_inputs['input_ids']
    token_type_ids = encoded_inputs['token_type_ids']

    if not is_test:
        label = np.array([example['label']], dtype='int64')
        return input_ids, token_type_ids, label
    else:
        qid = np.array([example['qid']], dtype='int64')
        return input_ids, token_type_ids, qid



# In[10]:


from utils import create_dataloader

max_seq_length=256
batch_size = 35

trans_func = partial(
    convert_example,
    tokenizer = tokenizer,
    max_seq_length = max_seq_length)

batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    stack()):[data for data in fn(samples)]

train_data_loader = create_dataloader(
    train_ds,
    mode = 'train',
    batch_size = batch_size,
    batchify_fn = batchify_fn,
    trans_fn = trans_func)

dev_data_loader = create_dataloader(
    dev_ds,
    mode = 'dev',
    batch_size = batch_size,
    batchify_fn=batchify_fn,
    trans_fn = trans_func)



# **模型训练**

# In[20]:


import time
from utils import evaluate

epochs = 10
ckpt_dir = 'skep_ckpt'
num_training_steps = len(train_data_loader) * epochs

optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())
criterion = paddle.nn.loss.CrossEntropyLoss()
metic = paddle.metric.Accuracy()

from visualdl import LogWriter
writer = LogWriter('./log')
best_val_acc = 0
global_step = 0
tic_train = time.time()
for epoch in range(1, epochs+1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()
        
        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:
            eval_loss, eval_accu = evaluate(model, criterion, metric, dev_data_loader)
            print('eval on dev loss:{:.8}'.format(eval_loss, eval_accu))

            writer.add_scalar(tag='eval/loss', step=global_step, value=eval_loss)
            writer.add_scalar(tag='eval/acc', step=global_step, value=eval_accu)

            writer.add_scalar(tag='train/loss', step=global_step, value=loss)
            writer.add_scalar(tag='train/acc', step=global_step, value=acc)
            save_dir = 'best_checkpoint'

            if eval_accu > best_val_acc:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                best_val_acc = eval_accu
                print(f"模型保存在 {global_step} 步， 最佳eval准确度为{best_val_acc:.8f}！")
                save_param_path = os.path.join(save_dir, 'best_model.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                fh = open('best_checkpoint/best_model.txt', 'w', encoding='utf-8')
                fh.write(f"模型保存在 {global_step} 步， 最佳eval准确度为{best_val_acc:.8f}！")
                fh.close()


# **测试数据**

# In[11]:


test_ds = load_dataset(read_test, data_path='测试集.csv', lazy=False)
test_ds = MapDataset(test_ds)
print(len(test_ds))


# In[ ]:


import numpy as np 
import paddle

trans_func = partial(
    convert_example,
    tokenizer = tokenizer,
    max_seq_length = max_seq_length,
    is_test = True)

batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    Stack()):[data for data in fn(samples)]

test_data_loader = create_dataloader(
    test_ds,
    model = 'test',
    batch_size = batch_size,
    batchify_fn = batchify_fn,
    trans_fn = trans_func)
    


# In[13]:


params_path = 'best_checkpoint/best_model.pdparams'
if params_path and os.path.isfile(params_path):
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print('Loaded parameters from %s' % params_path)


# In[14]:


lable_map = {0:'1', 1:'2', 2:'3', 3:'4', 4:'5'}
results = []
model.eval()
for batch in test_data_loader:
    input_ids, token_type_ids, qids =batch
    logits = model(input_ids, token_type_ids)
    probs = F.softmax(logits, axis=-1)
    idx = paddle.argmax(probs, axis=1).numpy()
    idx = idx.tolist()
    labels = [lable_map[i] for i in idx]
    qids = qids.numpy().tolist()
    results.extend(zip(qids, labels))


# **保存结果**

# In[15]:


with open('submission.csv', 'w', encoding='utf-8') as f:
    f.write('id, score\n')

    for (idx, label) in results:
        f.write('TEST_' + str(idx[0])+',' + label + '\n')


# In[16]:


get_ipython().system('tail 测试集.csv')


# In[17]:


get_ipython().system(' head submission.csv')


# In[18]:


get_ipython().system('tail submission.csv')

