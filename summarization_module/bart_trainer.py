import torch
import json
import os
import sys
import argparse

from tqdm import tqdm
from threading import Thread
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW, lr_scheduler

## config
batch_size = 1
model_name = 'gogamza/kobart-base-v2'
dataset_file = 'data/dataset.pt'
train_file = 'data/train_original.json' # num of total data is about 240000
valid_file = 'data/valid_original.json' # num of total data is about 30000
num_threads = 8

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# read json & tokenize
def get_input_and_labels(documents, articles, abstractives):
    for document in documents:
        article = ''
        for text in document['text']:
            if len(text) > 0:
                article += (text[0]['sentence'] + ' ')
        articles.append(article)
        
        abstractive = document['abstractive']
        if len(abstractive) > 0:
            abstractive = abstractive[0]
        abstractives.append(abstractive)
        
def get_dataset_from_json(json_file, num_data=0):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        documents = json_data['documents']
        data_size = len(documents)
        if num_data == 0 or num_data > data_size:
            num_data = data_size
        
        data_per_threads = num_data//num_threads
        t_results = []
        threads = []
        for i in range(num_threads):
            t_result = [[], []]
            t_results.append(t_result)
            
            thread = Thread(target=get_input_and_labels, args=(documents[i*data_per_threads:(i+1)*data_per_threads], t_result[0], t_result[1],))
            thread.daemon = True
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
        
        data_dict = {'article':[], 'abstractive':[]}
        for t_result in t_results:
            data_dict['article'].extend(t_result[0])
            data_dict['abstractive'].extend(t_result[1])
            
        return Dataset.from_dict(data_dict)



def preprocess(examples):
    inputs = tokenizer(examples['article'], return_tensors='pt', max_length=1024, padding='max_length', truncation=True)
    labels = tokenizer(examples['abstractive'], return_tensors='pt', max_length=1024, padding='max_length', truncation=True)
    inputs['labels'] = labels['input_ids']
    return inputs

def create_dataloader(dataset):
    input_ids = dataset['input_ids']
    attention_mask = dataset['attention_mask']
    labels = dataset['labels']
    tensor_dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(tensor_dataset, batch_size=batch_size)


# evaluate model
@torch.no_grad()
def eval_model(model, val_dataloader):
    device = next(model.parameters()).device
    model.to(device)
    model.eval()
    total_loss = 0
    
    print('=== evaluate model')
    for _, data in enumerate(tqdm(val_dataloader)):
        data = [t.to(device) for t in data]
        inputs = {
            'input_ids': data[0],
            'attention_mask': data[1],
            'labels': data[2]
        }
        outputs =  model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
    
    total_loss /= len(val_dataloader)
    print(f'total loss : {total_loss}')
    
    return total_loss



class Checkpoint():
    def __init__(self, model, optimizer, scheduler) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.last_step = -1
        self.best_loss = 1e20
        
    def set_root_dir(self, root_dir):
        if root_dir is not None:
            self.root_dir = root_dir
            self.path = os.path.join(root_dir, 'checkpoint.pt')
            
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
                
            if os.path.exists(self.path):
                self.load(self.path)
    
    def load(self, save_path):
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.last_step = checkpoint['last_step']
        self.best_loss = checkpoint['best_loss']
    
    def save(self):
        if not self.path is None:
            torch.save({
                'model' : self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'scheduler' : self.scheduler.state_dict(),
                'epoch' : self.epoch,
                'last_step' : self.last_step,
                'best_loss' : self.best_loss
            }, self.path)
        
    def step(self):
        self.optimizer.step()
        self.last_step += 1
    
    def eval(self, val_dataloader):
        if not self.root_dir is None:
            loss = eval_model(self.model, val_dataloader)
            if self.loss > loss:
                self.loss = loss
                torch.save(self.model.state_dict(), os.path.join(self.root_dir, 'best.pt'))
    
    def next(self):
        self.scheduler.step()
        self.epoch += 1
        self.last_step = -1
        self.save()
        
    def close(self):
        if not self.path is None and os.path.exists(self.path):
            os.remove(self.path)


def train_model(model, dataloader, checkpoint_dir=None, epochs=1, lr=2e-5, device=torch.device('cuda')):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:0.95**epoch)
    checkpoint = Checkpoint(model, optimizer, scheduler)
    checkpoint.set_root_dir(checkpoint_dir)

    for epoch in range(checkpoint.epoch, epochs):
        print(f'=== train model {epoch}/{epochs}')
        model.train()
        num_trained = 0
        total_loss = 0
        
        for step, data in enumerate(tqdm(dataloader['train'])):
            if step <= checkpoint.last_step:
                continue
            
            data = [t.to(device) for t in data]
            inputs = {
                'input_ids': data[0],
                'attention_mask': data[1],
                'labels': data[2]
            }

            # get loss
            optimizer.zero_grad()
            outputs =  model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            checkpoint.step()
            num_trained += 1
            
            # save checkpoint 
            if step % 1000 == 0:
                checkpoint.save()
                print(f'loss : {total_loss/num_trained}')
        
        checkpoint.eval(dataloader['val'])
        checkpoint.next()
        
    # remove checkpoint
    checkpoint.close()


def main(args):
    if bool(args.data):
        dataloader = torch.load(dataset_file)
    else:
        train_dataset = get_dataset_from_json(train_file)
        val_dataset = get_dataset_from_json(valid_file)
        dataloader = {
            'train': create_dataloader(train_dataset.map(preprocess, batched=True).with_format("torch")),
            'val': create_dataloader(val_dataset.map(preprocess, batched=True).with_format("torch"))
        }
        torch.save(dataloader, dataset_file)
        
        
    train_model(model, dataloader, './checkpoint', epochs=5)
    torch.save(model, 'bart.pt')
    model.push_to_hub('yeti-s/kobart-base-v2-news-summarization', use_auth_token=args.token)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train bart model & upload hugginface hub')
    parser.add_argument('--token', required=True, help='write token for huggingface hub')
    parser.add_argument('--data', required=False, default=False, help='load dataloader from dataset.pt')
    main(parser.parse_args())