# 竞赛系列：商品评论情感预测分析

电商平台中,很多用户都会基于自己的购物体验对商品进行评分和评论.但有些用户只给出了评论而没有评分，没有了评分的量化标准，这给商家进行数据运营与选品决策带来了困难。如何根据商品评论估计出相对应的评分，这是情感分析的问题，而得到这些信息，也有利于对应商品的生产自身竞争力的提高，以及为用户提供高质量感兴趣的商品。
我们将采用 PaddleNLP 对本数据集的数据进行情感分析，用 NLP 自然语言处理中常用的 Transformer 模型进行训练

数据简介

本数据集包括52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据
本次练习赛所使用数据集基于JD的电商数据，来自WWW的JD.com E-Commerce Data，并且针对部分字段做出了一定的调整，所有的字段信息请以本练习赛提供的字段信息为准
评分为[1,5] 之间的整数

二、数据初步处理

!pip install -U paddlenlp

1.解压数据
!tar -xvf data/data96333/商品评论情感预测.gz

**2.查看数据**
!head 训练集.csv

!head 测试集.csv

!head submission.csv

**3.重写read方法读取自定义数据集**


```python
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

```

**4.训练集载入**


```python
# data_path为read()方法的参数
dataset_ds = load_dataset(read, data_path='训练集.csv',lazy=False)
# 在这进行划分
train_ds = Subset(dataset=dataset_ds, indices=[i for i in range(len(dataset_ds)) if i % 10 != 1])
dev_ds = Subset(dataset=dataset_ds, indices=[i for i in range(len(dataset_ds)) if i % 10 == 1])

test_ds =  load_dataset(read_test, data_path='测试集.csv',lazy=False)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_97/1472104333.py in <module>
          1 # data_path为read()方法的参数
    ----> 2 dataset_ds = load_dataset(read, data_path='训练集.csv',lazy=False)
          3 # 在这进行划分
          4 train_ds = Subset(dataset=dataset_ds, indices=[i for i in range(len(dataset_ds)) if i % 10 != 1])
          5 dev_ds = Subset(dataset=dataset_ds, indices=[i for i in range(len(dataset_ds)) if i % 10 == 1])


    NameError: name 'load_dataset' is not defined



```python
for i in range(5):
    print(test_ds[i])
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_97/3996904181.py in <module>
          1 for i in range(5):
    ----> 2     print(test_ds[i])
    

    NameError: name 'test_ds' is not defined



```python
# 在转换为MapDataset类型
train_ds = MapDataset(train_ds)
dev_ds = MapDataset(dev_ds)
test_ds = MapDataset(test_ds)
print(len(train_ds))
print(len(dev_ds))
print(len(test_ds))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_97/596802951.py in <module>
          1 # 在转换为MapDataset类型
    ----> 2 train_ds = MapDataset(train_ds)
          3 dev_ds = MapDataset(dev_ds)
          4 test_ds = MapDataset(test_ds)
          5 print(len(train_ds))


    NameError: name 'MapDataset' is not defined


**三、SKEP模型加载**


```python
# 指定模型名称一键加载模型
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

model = SkepForSequenceClassification.from_pretrained(
    'skep_ernie_1.0_large_ch', num_classes=  5)
# 指定模型名称一键加载tokenizer
tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')
```

    [2022-02-26 13:10:21,478] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/skep/skep_ernie_1.0_large_ch.pdparams and saved to /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch
    [2022-02-26 13:10:21,521] [    INFO] - Downloading skep_ernie_1.0_large_ch.pdparams from https://paddlenlp.bj.bcebos.com/models/transformers/skep/skep_ernie_1.0_large_ch.pdparams
    100%|██████████| 1238309/1238309 [00:28<00:00, 42891.03it/s]
    [2022-02-26 13:11:22,287] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/skep/skep_ernie_1.0_large_ch.vocab.txt and saved to /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch
    [2022-02-26 13:11:22,290] [    INFO] - Downloading skep_ernie_1.0_large_ch.vocab.txt from https://paddlenlp.bj.bcebos.com/models/transformers/skep/skep_ernie_1.0_large_ch.vocab.txt
    100%|██████████| 55/55 [00:00<00:00, 24627.60it/s]


**四、数据NLP特征处理**


```python
import os
from functools import partial


import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad

from utils import create_dataloader

def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False):
   
    # 将原数据处理成model可读入的格式，enocded_inputs是一个dict，包含input_ids、token_type_ids等字段
    encoded_inputs = tokenizer(
        text=example["text"], max_seq_len=max_seq_length)

    # input_ids：对文本切分token后，在词汇表中对应的token id
    input_ids = encoded_inputs["input_ids"]
    # token_type_ids：当前token属于句子1还是句子2，即上述图中表达的segment ids
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        # label：情感极性类别
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        # qid：每条数据的编号
        qid = np.array([example["qid"]], dtype="int64")
        return input_ids, token_type_ids, qid
```


```python
from utils import create_dataloader
# 处理的最大文本序列长度
max_seq_length=256
# 批量数据大小
batch_size=35

train_ds = Subset(dataset=dataset_ds, indices=[i for i in range(len(dataset_ds)) if i % 10 != 1])
dev_ds = Subset(dataset=dataset_ds, indices=[i for i in range(len(dataset_ds)) if i % 10 == 1])


# 将数据处理成模型可读入的数据格式
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)

# 将数据组成批量式数据，如
# 将不同长度的文本序列padding到批量式数据中最大长度
# 将每条数据label堆叠在一起
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack()  # labels
): [data for data in fn(samples)]
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```

**五、模型训练**

**1.训练准备**


```python
import time

from utils import evaluate

# 训练轮次
epochs = 10
# 训练过程中保存模型参数的文件夹
ckpt_dir = "skep_ckpt"
# len(train_data_loader)一轮训练所需要的step数
num_training_steps = len(train_data_loader) * epochs

# Adam优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=2e-5,
    parameters=model.parameters())
# 交叉熵损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()
# accuracy评价指标
metric = paddle.metric.Accuracy()
```

**2.开始训练**


```python
# 开启训练

# 加入日志显示
from visualdl import LogWriter

writer = LogWriter("./log")
best_val_acc=0
global_step = 0
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch
        # 喂数据给model
        logits = model(input_ids, token_type_ids)
        # 计算损失函数值
        loss = criterion(logits, labels)
        # 预测分类概率值
        probs = F.softmax(logits, axis=1)
        # 计算acc
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
        
        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:
            # 评估当前训练的模型
            eval_loss, eval_accu = evaluate(model, criterion, metric, dev_data_loader)
            print("eval  on dev  loss: {:.8}, accu: {:.8}".format(eval_loss, eval_accu))
            # 加入eval日志显示
            writer.add_scalar(tag="eval/loss", step=global_step, value=eval_loss)
            writer.add_scalar(tag="eval/acc", step=global_step, value=eval_accu)
            # 加入train日志显示
            writer.add_scalar(tag="train/loss", step=global_step, value=loss)
            writer.add_scalar(tag="train/acc", step=global_step, value=acc)
            save_dir = "best_checkpoint"
            # 加入保存       
            if eval_accu>best_val_acc:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                best_val_acc=eval_accu
                print(f"模型保存在 {global_step} 步， 最佳eval准确度为{best_val_acc:.8f}！")
                save_param_path = os.path.join(save_dir, 'best_model.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                fh = open('best_checkpoint/best_model.txt', 'w', encoding='utf-8')
                fh.write(f"模型保存在 {global_step} 步， 最佳eval准确度为{best_val_acc:.8f}！")
                fh.close()
```

**六、预测提交结果**

**1.测试数据集处理**


```python

test_ds =  load_dataset(read_test, data_path='测试集.csv',lazy=False)
# 在转换为MapDataset类型
test_ds = MapDataset(test_ds)
print(len(test_ds))
```


```python
import numpy as np
import paddle

# 处理测试集数据
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    is_test=True)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack() # qid
): [data for data in fn(samples)]
test_data_loader = create_dataloader(
    test_ds,
    mode='test',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```

**2.加载预测模型**


```python

# 根据实际运行情况，更换加载的参数路径
params_path = 'best_checkpoint/best_model.pdparams'
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
```

**3.开始预测**


```python

# 处理测试集数据
label_map = {0: '1', 1:'2', 2:'3', 3:'4',4:'5'}
results = []
# 切换model模型为评估模式，关闭dropout等随机因素
model.eval()
for batch in test_data_loader:
    input_ids, token_type_ids, qids = batch
    # 喂数据给模型
    logits = model(input_ids, token_type_ids)
    # 预测分类
    probs = F.softmax(logits, axis=-1)
    idx = paddle.argmax(probs, axis=1).numpy()
    idx = idx.tolist()
    labels = [label_map[i] for i in idx]
    qids = qids.numpy().tolist()
    results.extend(zip(qids, labels))
```

**4.保存结果**


```python
# 写入预测结果
with open( "submission.csv", 'w', encoding="utf-8") as f:
    # f.write("数据ID,评分\n")
    f.write("id,score\n")

    for (idx, label) in results:
        f.write('TEST_'+str(idx[0])+","+label+"\n")
```

**5.检查结果**


```python
!tail 测试集.csv
```
