import torch
import json
import os
import argparse
import glob
import functools

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW, lr_scheduler

## config
model_name = 'yeti-s/kobart-title-generator'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)    

# read json & tokenize
def replace_special_char(text):
    text = text.replace("\\\'", "\'")
    text = text.replace('\\\"', '\"')
    return text.replace("\\", "")

def get_data_from_json(json_files):
    contents = []
    titles = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='UTF8') as f:
            json_data = json.load(f)
            source_data_info = json_data['sourceDataInfo']
            contents.append(replace_special_char(source_data_info['newsContent']))
            titles.append(replace_special_char(source_data_info['newsTitle']))
            
    return contents, titles


def create_dataset(data_dir, num_threads, num_data = 0):
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    num_jsons = len(json_files)
    if num_data == 0 or num_data > num_jsons:
        num_data = num_jsons
    json_files = json_files[:num_data]
    
    data_per_threads = num_data//num_threads
    with ThreadPoolExecutor(max_workers=num_threads-1) as executor:
        threads = [executor.submit(get_data_from_json, json_files[i*data_per_threads:(i+1)*data_per_threads]) for i in range(num_threads-1)]
        contents, titles = get_data_from_json(json_files[(num_threads-1)*data_per_threads:])
        
        for thread in threads:
            sub_contents, sub_titles = thread.result()
            contents.extend(sub_contents)
            titles.extend(sub_titles)
            
        data_dict = {'content':contents, 'title':titles}
        return Dataset.from_dict(data_dict)


def preprocess(examples, max_length):
    inputs = tokenizer(examples['content'], return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    labels = tokenizer(examples['title'], return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    inputs['labels'] = labels['input_ids']
    return inputs

def create_dataloader(dataset, batch_size):
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
    def __init__(self, model, optimizer, scheduler, root_dir, auth_token) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.last_step = -1
        self.best_loss = 1e20
        self.auth_token = auth_token
        self.set_root_dir(root_dir)
        
        
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
            if self.best_loss > loss:
                self.best_loss = loss
                torch.save(self.model.state_dict(), os.path.join(self.root_dir, 'best.pt'))
                
                if self.auth_token is not None:
                    self.model.push_to_hub(model_name, use_auth_token=self.auth_token)
    
    def next(self):
        self.scheduler.step()
        self.epoch += 1
        self.last_step = -1
        self.save()
        
    def close(self):
        if not self.path is None and os.path.exists(self.path):
            os.remove(self.path)


def train_model(model, dataloader, checkpoint_dir=None, auth_token=None, epochs=1, lr=2e-5, device=torch.device('cuda')):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:0.95**epoch)
    checkpoint = Checkpoint(model, optimizer, scheduler, checkpoint_dir, auth_token)
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
            if (step+1) % 1000 == 0:
                checkpoint.save()
                print(f'loss : {total_loss/num_trained}')
        
        checkpoint.eval(dataloader['val'])
        checkpoint.next()
        
    # remove checkpoint
    checkpoint.close()


def main(args):
    if args.dataset is not None:
        dataloader = torch.load(args.dataset)
        
    elif args.train is not None and args.val is not None:
        train_dataset = create_dataset(args.train, int(args.num_threads))
        val_dataset = create_dataset(args.val, int(args.num_threads))
        preprocessor = functools.partial(preprocess, max_length=int(args.max_length))
        dataloader = {
            'train': create_dataloader(train_dataset.map(preprocessor, batched=True).with_format("torch"), int(args.batch_size)),
            'val': create_dataloader(val_dataset.map(preprocessor, batched=True).with_format("torch"), int(args.batch_size))
        }
        
        if args.save_dataset is not None:
            torch.save(dataloader, args.save_dataset)
            
    else:
        print("'--dataset' or '--train and --val' needed!")
        exit(1)

    train_model(model, dataloader, checkpoint_dir=args.checkpoint, auth_token=args.token, epochs=5)
    torch.save(model, 'last.pt')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train bart model & upload hugginface hub')
    parser.add_argument('--token', required=False, default=None, help='write token for huggingface hub')
    parser.add_argument('--dataset', required=False, default=None, help='load dataloader from dataset.pt')
    parser.add_argument('--train', required=False, default=None, help='train json files dir path')
    parser.add_argument('--val', required=False, default=None, help='validation json files dir path')
    parser.add_argument('--save_dataset', required=False, default=None, help='save created dataset from train and val data')
    parser.add_argument('--max_length', required=False, default=764, help='max token size')
    parser.add_argument('--checkpoint', required=False, default=None, help='train checkpoint dir')
    parser.add_argument('--num_threads', required=False, default=1, help='num threads for creating dataset from json files')
    parser.add_argument('--batch_size', required=False, default=1, help='batch size')
    main(parser.parse_args())