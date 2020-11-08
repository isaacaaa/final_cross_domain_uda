import torch.nn as nn
import json
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
# from load_data import prepare_source,prepare_evaluate,prepare_data,load_obj,save_obj
import os,sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from domain_similarity import compute_psi_for_test
# from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import argparse
import glob
import random
import logging
import os
from scipy.special import softmax
import random
from torch import optim
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
device = torch.device("cuda")
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from x_data_processor import appreview_processors as processors
from x_data_processor import appreview_output_modes as output_modes
from sklearn import metrics
from transformers import BertTokenizer, BertModel
from sklearn import metrics
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
 
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    get_linear_schedule_with_warmup,
)
from modeling_bert_share import BertForSequenceClassification

# from new_modeling_bert import BertForSequenceClassification
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(42)

def confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred, range(len(processors['appreview']().get_labels())))

def basic_metrics(y_true, y_pred):
    return {'Accuracy': metrics.accuracy_score(y_true, y_pred),
            'Precision': metrics.precision_score(y_true, y_pred, average='macro'),
            'Recall': metrics.recall_score(y_true, y_pred, average='macro'),
            'Macro-F1': metrics.f1_score(y_true, y_pred, average='macro'),
            'Micro-F1': metrics.f1_score(y_true, y_pred, average='micro'),
            'ConfMat': confusion_matrix(y_true, y_pred)}

def init_w(embedding_dim):
    w = torch.Tensor(embedding_dim,1)
    # w= torch.Tensor(embedding_dim,embedding_dim)
    return nn.Parameter(nn.init.xavier_uniform_(w).to(device)) # sigmoid gain=1
def get_source_instance(data_dir,evaluate):
    
    # model.to(device)
    # tokenizer.to('cuda')
    processor=processors['appreview']()
    unsup_processor=processors['unsup']()
    # processor=processors['appreview']()
    # processor=processors['appreview']()
    if evaluate==True:
        examples1,examples2,examples3,examples4 = (
            processor.get_train_examples(data_dir)
        )
        # book,b_label=get_bert_embedding(examples1)
        
        # dvd,d_label=get_bert_embedding(examples2)
        # electronic,e_label=get_bert_embedding(examples3)
        # kitchen,k_label=get_bert_embedding(examples4)
        t_examples1,t_examples2,t_examples3,t_examples4 = (
            processor.get_test_examples(data_dir)
        )
        # t_book,t_book_label=get_bert_embedding(t_examples1)
        # t_dvd,t_dvd_label=get_bert_embedding(t_examples2)
        # t_electronics,t_electronic_label=get_bert_embedding(t_examples3)
        t_kitchen,t_kitchen_label=get_bert_embedding(t_examples4)
        # return book,b_label,dvd,d_label,electronic,e_label,kitchen,k_label,t_book,t_book_label,t_dvd,t_dvd_label,t_electronics,t_electronic_label,t_kitchen,t_kitchen_label
        return t_kitchen,t_kitchen_label
    else:
        examples1,examples2,examples3,examples4 = (
            processor.get_train_examples(data_dir)
            )
        book,b_label=get_bert_embedding(examples1)
        dvd,d_label=get_bert_embedding(examples2)     
        electronic,e_label=get_bert_embedding(examples3)
        # kitchen,k_label=get_bert_embedding(examples4[:32])
        
        # unsup_examples1,unsup_examples2,unsup_examples3,unsup_examples4 = (
        #     unsup_processor.get_train_examples(data_dir)
        # )
        # u_book,x=get_bert_embedding(unsup_examples1)
        # u_dvd,y=get_bert_embedding(unsup_examples2)
        # u_electronic,z=get_bert_embedding(unsup_examples3)
        # u_kitchen,w=get_bert_embedding(unsup_examples4[:32])
        return book,b_label,dvd,d_label,electronic,e_label
        # return book,b_label,dvd,d_label,electronic,e_label,kitchen,k_label
    # for i in enumerate(examples)
def get_bert_embedding(examples):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased',return_dict=True)
    label_list=['positive','negative']
    features1 = convert_examples_to_features(
        examples, tokenizer, max_length=256, label_list=label_list, output_mode='classification',
    )
    all_input_ids1= torch.tensor([f.input_ids for f in features1], dtype=torch.long)
       
    all_attention_mask1 = torch.tensor([f.attention_mask for f in features1], dtype=torch.long)
    all_token_type_ids1 = torch.tensor([f.token_type_ids for f in features1], dtype=torch.long)
    all_labels1 = torch.tensor([f.label for f in features1], dtype=torch.long)
    books_dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels1)
    train_sampler = SequentialSampler(books_dataset)
    train_dataloader = DataLoader(books_dataset, sampler=train_sampler, batch_size=16)
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
    book=[]
    label=[]
    for step, batch in enumerate(epoch_iterator):
            model.to(device)
            # model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],"labels":batch[3]}
            label.append(inputs["labels"])
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2]}
            # print(inputs)
            outputs = model(**inputs)
            # print(outputs.pooler_output.shape)
            book.append(outputs.pooler_output.detach().cpu())
            
    return book,label
    # return books_dataset
book,b_label,dvd,d_label,electronic,e_label =get_source_instance("sss",False)
# t_kitchen,t_kitchen_label=get_source_instance("sss",True)
# print(b_label)
# print(len(b_label))
# print(b_label[0].shape)

# print(b_label.shape)
def reshape(book):

    # print(book)
    # print(len(book))
    for i in book:
        continue
        # print(i.shape)
        # print(i.shape[1])
        # print(i.shape[2])
    # book=torch.tensor(book)
    book= torch.stack(book, 0)
    x=book.shape[0]
    y=book.shape[1]#16=number of instance
    z=book.shape[2]#768=embedding_dim
    book=book.view(x*y,768)
    return book
def reshape_label(b_label):
    b_label=torch.stack(b_label,0)
    x=b_label.shape[0]
    y=b_label.shape[1]

    b_label=b_label.view(-1,x*y)
    b_label=b_label.squeeze(0)
    # b_label=b_label.reshape(b_label)
    # print('xxxxxxxxxxxxxxxxxxxxxxxx')
    # print(b_label)
    # print(b_label.shape)
    return b_label

b_label=reshape_label(b_label)
d_label=reshape_label(d_label)
e_label=reshape_label(e_label)
# k_label=reshape_label(k_label)
cat_label=torch.cat((b_label,d_label),0)
cat_label=torch.cat((cat_label,e_label),0)
# cat_label=torch.cat((cat_label,k_label),0)
#print('all_y')
#print(cat_label)
#print(cat_label.shape)

book=reshape(book)
# print(book.shape)
dvd=reshape(dvd)
electronic=reshape(electronic)
# kitchen=reshape(kitchen)

cat=torch.cat((book,dvd),0)
# print(cat.shape)
cat=torch.cat((cat,electronic),0)
cat=cat.to(device)
# cat=torch.cat((cat,kitchen),0)
# print(cat.shape)

# u_book=reshape(u_book)
# u_dvd=reshape(u_dvd)
# u_electronic=reshape(u_electronic)
# u_kitchen=reshape(u_kitchen)
# u_cat=torch.cat((u_book,u_dvd),0)
# u_cat=torch.cat((u_cat,u_electronic),0)
# u_cat=u_kitchen
def cos_sim(a,b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) if (np.linalg.norm(a)*np.linalg.norm(b))!=0 else 0
    return cos_sim
def compute_centriod(instances):
    a = np.array(instances)
    print(a.shape)
    # a = instances
    print (np.mean(a,axis=0).shape)
    
    return np.mean(a,axis=0)

def unlabel_sim(tgt_un):
    # tgt_un = load_obj("%s/X_un"%target)
    # print target,tgt_un.shape
    filter_u_cat=[]
    c_t = compute_centriod(tgt_un)
    computed_tgt_sim = [cos_sim(x,c_t) for x in tgt_un]
    print('simsimsimsismismismismism')
    print(computed_tgt_sim)
    topk=sorted(range(len(computed_tgt_sim)),key=lambda i:computed_tgt_sim[i])[-5000:]
    for i in topk:
        filter_u_cat.append(tgt_un[i])
    #   print(cos_sim(i,c_t))
      #  if cos_sim(i,c_t)>statistics.mean(computed_tgt_sim):
       #     filter_u_cat.append(i)
            
    return filter_u_cat,topk
# u_cat,topk=unlabel_sim(u_cat.numpy())
# print(u_cat)
# print(len(u_cat))
# u_cat=np.array(u_cat)
# u_cat=torch.from_numpy(u_cat)
# print(u_cat)
# print(u_cat.shape)

# print('all_x')
# print(u_cat)
# print(u_cat.shape)

# u_cat=torch.cat((u_cat,u_kitchen),0)
# t_book=reshape(t_book)
# t_dvd=reshape(t_dvd)
# t_electronic=reshape(t_electronic)
# t_kitchen=reshape(t_kitchen)
# u_cat=torch.cat((u_book,u_dvd),0)
# u_cat=torch.cat((u_cat,u_electronic),0)

# t_kitchen_label=reshape_label(t_kitchen_label)
# u_dvd=reshape(u_dvd)
# u_electronic=reshape(u_electronic)
# u_kitchen=reshape(u_kitchen)
# u_cat=torch.cat((u_book,u_dvd),0)
# u_cat=torch.cat((u_cat,u_electronic),0)


class DomainAttention(nn.Module):
    # embedding_dim: embedding dimensionality
    # hidden_dim: hidden layer dimensionality
    # source_size: number of source domains
    # hidden_dim, batch_size, label_size, ,num_instances=1600
    def __init__(self, embedding_dim, source_size, y,label_size=2):
        super(DomainAttention, self).__init__()
        # bias
        self.bias = nn.Parameter(torch.Tensor([0]).to(device))
        # source phi, multi sources, just like python list
        self.phi_srcs = nn.ParameterList([init_w(embedding_dim) for i in range(source_size)])
        # labels for source domain (y(d_j))
        self.y = y
        self.label_size = label_size
        self.sigmoid = nn.Sigmoid()
        self.source_size = source_size
        # self.reshape=reshape()


    def forward(self, batch_x,cat):
        # batch_x= torch.stack(batch_x, 0)
        # x=batch_x.shape[0]
        # y=batch_x.shape[1]#16=number of instance
        # z=batch_x.shape[2]#768=embedding_dim
        # batch_x=batch_x.view(x*y,768)
        # batch_x=batch_x.cpu().detach()
        print(batch_x.shape)
        # batch_x=reshape(batch_x)
        all_y_hat=[]
        
        x=batch_x.to(device)

        ###
        # x:target instance
        ###
        # print('model_x')
        # print(x)
        # print(x.shape)
        # x = x.view(-1,len(x))
        # print(x.shape)
        # print('model_x')
        # print(x)
        # print(x.shape)
        # cat=cat.to(device)
        y = self.y.view(-1,len(self.y)).to(device)
        # print(y.shape)#1,96#
        # print(cat.shape)
        # print(x)
        # print(x.shape)
        # print(cat)
        # print(cat.shape)
        
        #sss=np.dot(cat,x.T)
        #sss=np.exp(sss)
        # sss=sss.deatch().cpu().clone().numpy()

        # print(sss)
        # print(sss.shape)
        # print(normalize(sss,axis=0))
        # print(normalize(sss,axis=0).shape)
        # print(softmax(normalize(sss,axis=0)))#96*1#
        #print('sss')
        
        #sss=softmax(normalize(sss,axis=0))
        #print(sss)
        #sss=softmax(sss)
        #print(sss)
        #print(sss.shape)
        #print('another')
        con=[]
        psisum=0
        # cat=torch.from_numpy(cat)
        print(cat[0].shape)
        for i in cat:
            # print('---------------')

            # i=i.view(1,len(i))
            # print(i.shape)
            # print(i)
            # print(i.shape)
            # print(x)
            # print(x.shape)
            i=torch.unsqueeze(i,0)
            # print(i)
            # print(i.shape)
            # print(x)
            # print(x.shape)
            tmp=torch.mm(i,x.T)
            # print(tmp)
            # print(tmp.shape)
            # tmp=np.dot(i,x.T)
            # psisum+=tmp
            
            con.append(tmp)
        # print(con)
        print(len(con))
        # con=np.expand_dims(np.array(con),axis=0)
        con= torch.stack(con, 0)
        shapex=con.shape[0]
        # y=con.shape[1]
        # z=con.shape[2]
        print(con)
        print(con.shape)
        con= con.view(shapex, -1)
        print(con)
        print(con.shape)
        # psi_splits=np.exp(con)/sum(np.exp(con))
        # con=normalize(con,axis=0)
        # print(con)
        con=F.normalize(con,p=4,dim=0)
        print(F.normalize(con,p=4,dim=0))
        # con=normalize(con,axis=1)
        print(con)
        # print(con.shape)
        #print(sss==con)
        m=nn.Softmax(dim=0)
        
        con=m(con).to(device)
        
        print(con)
        print(con.shape)
        print(torch.sum(con))
        
        # b=c
        # con=np.round(con,4)
        # print(con)
        # print(con.shape)
        # xxxxxxxxxxxxxxxxxxx=xxx
        # psi_splits=con
        # psi_splits=torch.from_numpy(con).to(device)
        # con =torch.from_numpy(con)
        
        # print('yyyyyyyyyyyyyyyyyyyyyyyyyyyy')
        # print(sss)
        #sss=torch.from_numpy(sss)
        # print(sss)
        psi_splits = torch.chunk(con,self.source_size,dim=0)
        print(psi_splits)
        print(len(psi_splits))
        print(psi_splits[0].shape)
        
        # print(psi_splits[0].shape)#32,1#
        # print(y)
        # print(y.shape)
        y = torch.chunk(y,self.source_size,dim=1)
        print(y)
        print(len(y))
        print(y[0].shape)
        # print(y)
        # print(y.shape)
        # print(psi_matrix)
        # print(psi_matrix.shape)
        # print(type(psi_matrix))
        # psi_matrix=psi_matrix.flatten()
        
        # psi_matrix = psi_matrix.view(-1,len(psi_matrix)).to(device)
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        
        # print(type(psi_matrix))
        # psi_splits = torch.chunk(psi_matrix,self.source_size,dim=1)
        # print(psi_splits)
        # print()
        # print(psi_splits[0].shape)
        # print(np.ndim(psi_splits[1]))
        # print(np.ndim(psi_splits[2]))
        # print('model_y')
        # print(y)
        # print(y.shape)

        # y_splits = torch.chunk(y,self.source_size,dim=1)
        # print(y_splits[0].shape)
        # print(y_splits)
        # get the sum of x * phi_src
        theta_splits = []
        sum_src = 0.0
        theta_splits1 = []
        for phi_src in self.phi_srcs:
            x=x.to(device)
            print(x.shape)
            print(phi_src.shape)
            temp = torch.exp(torch.mm(x,phi_src))
            # temp1=torch.mm(x,phi_src)
            print(temp)
            print(temp.shape)
            # temp = torch.tensor([[0.]]).to(device) if torch.isnan(temp)==True else temp
            # prod = torch.mm(x,phi_src)
            # temp = torch.exp(prod)
            # temp = torch.mm(x,phi_src)
            theta_splits.append(temp)
            # theta_splits1.append(temp1)
            # print('templ')
            # print(temp)
            sum_src+=temp
            # print temp,torch.sum(x),torch.sum(phi_src)
        # theta_splits1=torch.FloatTensor(theta_splits1).to(device)
        # print(theta_splits1)
        # print(type(theta_splits1))  
        # theta_splits1=nn.Softmax(theta_splits1)
        # print(torch.sum(theta_splits1))
        # print(theta_splits)
        # d=sa
        # print(theta_splits[0].shape)
        # theta_splits=torch.FloatTensor(theta_splits).to(device)
        
        print(theta_splits)
        print(sum_src)
        #########
        # theta_splits=torch.div(theta_splits,sum_src)
        #x=nn.Softmax(dim=1)
        ###########
        # print(theta_splits)
        # x=nn.Softmax(dim=0)
        # theta_splits=x(theta_splits)
        
        # print(theta_splits)
        # print(theta_splits.shape)
        sum_matrix = 0.0
        count = 0
        # theta_splits=theta_splits.squeeze()
        # print(theta_splits)
        print(theta_splits[0])
        print(theta_splits[1])
        print(theta_splits[2])
        
        # print(len(theta_splits))
        # print(len(y_splits))
        # print(len(theta_splits))
        # print(theta_splits[0].shape)
        
        print(len(psi_splits))
        print(len(y))
        print(len(theta_splits))
        for theta,psi_split,y_split in zip(theta_splits,psi_splits,y):
            count += 1
            theta_matrix = theta/sum_src
            # psi_split=psi_split/psisum
            # print(y_split)
            # print(psi_split)
            # print(theta_matrix)
            # temp = y_split*psi_split*theta_matrix
            # print(y_splits)
            # print(theta_splits)
            # print('modeldmode;model')
            # print(psi_matrix)
            # print(psi_matrix.shape)
            # print(y_split)
            # print(y_split.shape)
            # print(psi_split)
            # print(psi_split.shape)
            # print(theta_matrix)
            # print(theta_matrix.shape)
            y_split=y_split.to(device).float()
            psi_split=psi_split.to(device).float()
            theta=theta.to(device).float()
            
            #theta_matrix= theta_matrix.to(device)
            
            # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            print(y_split.shape)
            print(psi_split.shape)
            a=y_split.T*psi_split
            # c=torch.mm(y_split,psi_split.T)
            # e=y_split*(psi_split.T)
            print(a)
            print(a.shape)
            # print(c)
            # print(c.shape)
            # print(e)
            # print(e.shape)
            # print(torch.sum(e))
            # print(y_split)
            # print(psi_split)
            # print(theta)
        
            #b=a*theta
            #print(b)
            #print(b.shape)
            print(theta)
            print(theta.shape)
            print(theta_matrix)
            print(theta_matrix.shape)
            # print(type(theta_matrix))
            # temp = y_split*psi_split*theta_matrix
            temp=torch.mul(a,theta.T)
            
            print('temp')
            print(temp)
            print(temp.shape)
            print(torch.sum(temp,0))
            print(torch.sum(temp,0).shape)
            sum_matrix += torch.sum(temp,0)
            print(sum_matrix)
            print(sum_matrix.shape)
            # print(self.bias)
            # print(sum_matrix)            # b=d
        print(self.bias)
        sum_matrix = sum_matrix  + self.bias
        # m=nn.Softmax(dim=0)
        print(sum_matrix)
        y_hat = self.sigmoid(sum_matrix)
        print('y_hat')
        print(y_hat)
        
        all_y_hat.append(y_hat)
        # pred.append(y_hat)
        # print(pred)
        return all_y_hat
config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
config = config_class.from_pretrained('bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained('bert-base-uncased',do_lower_case=True)
model = model_class.from_pretrained('bert-base-uncased')
# model.classifier1= nn.Linear(config.hidden_size, config.num_labels)
model.classifier2=DomainAttention(embedding_dim = 768, 
                           source_size = 3,
                           y =cat_label)

optimizer = optim.Adam(model.parameters(),lr=0.00002)
loss_function=nn.BCELoss(reduction='mean')
# optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
# scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
# print(model)
# print(model.parameters())

# for name, param in model.named_parameters():
#     if param.requires_grad:
        # print (name, param.data)

# def train(model,X_train,y_train,cat,loss_function,optimizer,i,rescale):
def train(model,labelled_kitchen,u_kitchen,y_train,cat,loss_function,optimizer,i,rescale):
    embedding_set=[]
    embedding_label_set=[]
    label_list=['positive','negative']
    # print(X_train.shape)
    # print(y_train.shape)
    # print(cat.shape)
    model.train()
    #labelled feature
    sup_features1 = convert_examples_to_features(
        labelled_kitchen, tokenizer, max_length=256, label_list=label_list, output_mode='classification',
    )
    sup_all_input_ids1= torch.tensor([f.input_ids for f in sup_features1], dtype=torch.long)
       
    sup_all_attention_mask1 = torch.tensor([f.attention_mask for f in sup_features1], dtype=torch.long)
    sup_all_token_type_ids1 = torch.tensor([f.token_type_ids for f in sup_features1], dtype=torch.long)
    sup_all_labels1 = torch.tensor([f.label for f in sup_features1], dtype=torch.long)
    #unlabelled feature
    features1 = convert_examples_to_features(
        u_kitchen, tokenizer, max_length=256, label_list=label_list, output_mode='classification',
    )
    all_input_ids1= torch.tensor([f.input_ids for f in features1], dtype=torch.long)
       
    all_attention_mask1 = torch.tensor([f.attention_mask for f in features1], dtype=torch.long)
    all_token_type_ids1 = torch.tensor([f.token_type_ids for f in features1], dtype=torch.long)
    all_labels1 = torch.tensor([f.label for f in features1], dtype=torch.long)
    # print(all_labels1)
    # print(y_train)
    
    all_labels1= y_train
    batch_size=16
    def slice(feature):
        concate=[]
        for index,x,y in enumerate(zip(sup_feature,unsup_feature)):
            concate.append()

    label_dataset = TensorDataset(sup_all_input_ids1, sup_all_attention_mask1, sup_all_token_type_ids1, sup_all_labels1)
    unlabel_dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels1)
    # print(model)
    # print(type(label_dataset))
    # print(label_dataset)
    # print(len(label_dataset))
    # all_dataset=label_dataset+unlabel_dataset
    # print(len(all_dataset))
    # print(all_dataset)
    

    sup_sampler = RandomSampler(label_dataset)
    unsup_sampler = RandomSampler(unlabel_dataset)
    
    sup_dataloader = DataLoader(label_dataset, sampler=sup_sampler, batch_size=8)
    sup_iterator = tqdm(sup_dataloader, desc="Iteration", disable=False)
    unsup_dataloader = DataLoader(unlabel_dataset, sampler=unsup_sampler, batch_size=16)
    unsup_iterator = tqdm(unsup_dataloader, desc="Iteration", disable=False)
    
    avg_loss=0.0
    sup_list=[]
    for step ,batch in enumerate(sup_iterator):
        batch = tuple(t.to(device) for t in batch)
        # print(bact)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],"labels":batch[3],'cat':cat,'return_dict':True,'classifier':'classifier1'}
        sup_list.append(inputs)
    for step, batch in enumerate(unsup_iterator):

        output_list=[]
        model.zero_grad()    
        model.to(device)
        # model.train()
        batch = tuple(t.to(device) for t in batch)
        # print(bact)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],"labels":batch[3],'cat':cat,'return_dict':True,'classifier':'classifier2'}
        # print(inputs)
        mod = step % len(sup_iterator)
        unsup_outputs = model(**inputs)
        sup_outputs=model(**sup_list[mod])
        
        
        print(sup_outputs)
        # print(outputs)
        # print(outputs[0][0])
        # print(label)
        # print(len(label))
        truth=batch[3].float()
        # print(truth)
        # truth=torch.tensor([label[step]]).float()
        # print(truth)
        # print(outputs)
        # for  i in outputs[0]:
            # output_list.append(i)
        # print(torch.tensor(output_list))
        # print(outputs[0])
        print('lllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll')
        print(unsup_outputs)
        print(unsup_outputs['logits'][0])
        print(truth)
        # output_list=torch.tensor(output_list,requires_grad=True).to(device)
        # print(output_list)
        optimizer.zero_grad()

        unsup_loss = loss_function(unsup_outputs['logits'][0],truth.to(device))
        weights=0.3
        sup_loss=sup_outputs[0]
        loss=sup_loss+weights*unsup_loss
        avg_loss+=loss
        loss.backward()#retain_graph=True
        optimizer.step()

        # model.zero_grad()
        
        # print(pred)
        # print truth_res[1032],pred_res[1032]
        
    # torch.save(model,'target.model')
        
    return avg_loss

            
def evaluate_epoch( model,test_examples, cat, loss_function,classifier):
    # model=torch.load('target.model')
    model.eval()
    label_list=['positive','negative']
    features1 = convert_examples_to_features(
        test_examples, tokenizer, max_length=256, label_list=label_list, output_mode='classification',
    )
    all_input_ids1= torch.tensor([f.input_ids for f in features1], dtype=torch.long)
       
    all_attention_mask1 = torch.tensor([f.attention_mask for f in features1], dtype=torch.long)
    all_token_type_ids1 = torch.tensor([f.token_type_ids for f in features1], dtype=torch.long)
    all_labels1 = torch.tensor([f.label for f in features1], dtype=torch.long)
    test_dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels1)
    # print(model)
    eval_sampler = SequentialSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=16)
    epoch_iterator = tqdm(eval_dataloader, desc="Iteration", disable=False)
    truth_res = []
    pred_list=[]
    preds_1=None
    out_label_ids=None
    # pred_res = []
    ori_pred=[]
    print('Start evaluating!!!')
    
    for step, batch in enumerate(epoch_iterator):
        output_list=[]
        model.to(device)
        # model.train()
        batch = tuple(t.to(device) for t in batch)
        if classifier=='classifier2':
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],'cat':cat,'return_dict':True,'classifier':classifier}
            
                outputs = model(**inputs)
                
                truth=batch[3].float().detach()
                
                print(outputs)
                print(outputs['logits'][0])
                pred=outputs['logits'][0]
                
                pred=pred.tolist()
                
                for index,i in enumerate(pred):
                    if i >0.5:
                        pred[index]=1       
                    else:
                        pred[index]=0
                pred_list=pred_list+pred
        else:
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2],'labels':batch[3],'cat':cat,'return_dict':True,'classifier':classifier}
            
                truth=batch[3]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                print(outputs)
                print(logits)
                # eval_loss += tmp_eval_loss.mean().item()
            
                # if preds_1 is None:
                preds_1 = logits.detach().cpu().numpy()
                # print('xxx')
                # print(preds_1)
                    # out_label_ids = inputs["labels"].detach().cpu().numpy()
                # else:
                    # print('yyy')
                    # print(preds_1)
                    # print(logits.detach().cpu().numpy())
                # preds_1 = np.append(preds_1, logits.detach().cpu().numpy(), axis=0)
                    # out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            # eval_loss = eval_loss / nb_eval_steps
            
                softmax_preds= np.argmax(preds_1, axis=1)
                # print(preds_1)
                preds_1_list=softmax_preds.tolist()
                truth_res+=truth.tolist()
                # out_label_ids_list=out_label_ids.tolist()
                pred_list=pred_list+preds_1_list        
            # result = compute_metrics(eval_task, preds, out_label_ids)
        
    
    from sklearn.metrics import accuracy_score
    print(pred_list)
    # print(all_labels1.tolist())
    print(truth_res)    
    results =basic_metrics(truth_res,pred_list)
    # score=accuracy_score(all_labels1.tolist(),pred_list)
    
    print('eval_results:',results)
    # print(metrics)
    # print('test avg_loss:%g acc:%g' % (avg_loss, acc))
    
    return results
# psi_matrix=[[1],[1],[1]]
# psi_matrix_test=compute_psi_for_test(cat,t_kitchen)
# print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
# print(psi_matrix_test.shape)
# psi_matrix_test=torch.tensor(psi_matrix_test).to(device)

EPOCH=5
best_f1=0.0
x=EPOCH-1
result_list=[]
y_star=np.load('unlabelled_psuedo_label.npy')
y_star=y_star[:10000]
# filtered_y_star=[]
# y_star=y_star.tolist()[:32]
# for k  in topk:
#     filtered_y_star.append(y_star[k])
    

# y_star=np.array(filtered_y_star)
y_star=torch.from_numpy(y_star)
# train_dataset = TensorDataset(u_cat,y_star)
# eval_dataset = TensorDataset(t_kitchen,t_kitchen_label)
# train_dataloader = DataLoader(train_dataset, sampler=None, batch_size=16)
# eval_dataloader = DataLoader(eval_dataset, sampler=None, batch_size=32)

# for x in range(s):
    # for batch_x,batch_y in train_dataloader:
        # i=0
        # i=i+1
        # print(x,i,batch_x,batch_y)
import os.path
from os import path
if path.exists('target_results.txt')==True:
    os.remove('target_results.txt')
for i in range(EPOCH):
    # model.train()
    
    unsup_processor=processors['unsup']()
    unsup_examples1,unsup_examples2,unsup_examples3,unsup_examples4 = (
            unsup_processor.get_train_examples('sss')
        )
    
    processor=processors['appreview']()
    examples1,examples2,examples3,examples4 = (
            processor.get_train_examples('sss')
        )
    t_examples1,t_examples2,t_examples3,t_examples4 = (
            processor.get_test_examples('sss')
        )
    
    loss=train(model,examples4,unsup_examples4[:10000],y_star,cat,loss_function,optimizer,i,0)
    results=evaluate_epoch(model,t_examples4,cat,loss_function,'classifier1')
    result_list.append(results)
    if results['Macro-F1']>best_f1:
        best_f1=results['Macro-F1']
        print('best_epoch:',i+1)
        print('best_epoch_F1:',best_f1)
        torch.save(model.state_dict(),'target.model')
    if i ==x:
        with open('target_results.txt','w') as f:
            f.write(str(best_f1))
            f.write('\n')
            f.write(str(result_list))
print(best_f1)
