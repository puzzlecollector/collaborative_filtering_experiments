import numpy as np 
import pandas as pd 
import os 
import time 
from pytorch_metric_learning import miners, losses, distances 
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler 
import torch 
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from tqdm.auto import tqdm 
import faiss # faiss-cpu 

apply_train_df = pd.read_csv("apply_train.csv") 

# candidate related information 
resume_certificate = pd.read_csv("resume_certificate.csv")
resume_education = pd.read_csv("resume_education.csv") 
resume_language = pd.read_csv("resume_language.csv") 

# company related information 
recruitment_df = pd.read_csv("recruitment.csv") 
company_df = pd.read_csv("company.csv") 

tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")

candidate_df = pd.merge(resume_certificate, resume_education, on='resume_seq', how='outer')
candidate_df = pd.merge(candidate_df, resume_language, on='resume_seq', how='outer')
candidate_df.fillna(-1, inplace=True)

candidate_numerical_cols = ["hischool_type_seq", "hischool_location_seq", "univ_type_seq1", "univ_type_seq2", "univ_transfer", "univ_location", "univ_major_type", "univ_score", "language", "exam_name", "score"]

candidate_str_cols = ["resume_seq", "certificate_contents", "hischool_special_type", "hischool_nation", "hischool_gender", "univ_major", "univ_sub_major"]

str_df = candidate_df[candidate_str_cols] 
numerical_df = candidate_df[candidate_numerical_cols] 

candidate_str_x = [] 
for i in tqdm(range(str_df.shape[0])): 
    cur_str = "" 
    for idx, col in enumerate(str_df.columns):
        cur_str += f"{col}: {str_df[col].values[i]}" + "[SEP]" 
    candidate_str_x.append(cur_str)
    
candidate_input_ids, candidate_attn_masks = [], [] 

for i in tqdm(range(len(candidate_str_x)), position=0, leave=True):
    encoded_inputs = tokenizer(candidate_str_x[i], max_length=156, truncation=True, padding="max_length")
    candidate_input_ids.append(encoded_inputs["input_ids"]) 
    candidate_attn_masks.append(encoded_inputs["attention_mask"]) 
    
candidate_input_ids = torch.tensor(candidate_input_ids, dtype=int) 
candidate_attn_masks = torch.tensor(candidate_attn_masks, dtype=int) 

print(candidate_input_ids.shape, candidate_attn_masks.shape) 

candidate_numerical_inputs = numerical_df.values 
candidate_numerical_inputs = torch.tensor(candidate_numerical_inputs).float() 

print(candidate_numerical_inputs.shape) 

resume_seqs = candidate_df["resume_seq"].values 

candidate_dict = {} 
for seq in resume_seqs:
    candidate_dict[seq] = {"input_ids":[], "attn_masks":[], "numerical_inputs":[]}
    
for i in tqdm(range(len(candidate_numerical_inputs)), position=0, leave=True):
    candidate_dict[resume_seqs[i]]["input_ids"] = candidate_input_ids[i] 
    candidate_dict[resume_seqs[i]]["attn_masks"] = candidate_attn_masks[i] 
    candidate_dict[resume_seqs[i]]["numerical_inputs"] = candidate_numerical_inputs[i] 
    
    
# company related data preprocessing 
company_df = pd.merge(recruitment_df, company_df, on='recruitment_seq', how='outer')

company_df.fillna(-1, inplace=True)

company_numerical_cols = ["address_seq1", "address_seq2", "address_seq3", "career_end", "career_start", "education", "major_task", "qualifications", "company_type_seq", "supply_kind", "employee"] 

company_str_cols = ["recruitment_seq", "check_box_keyword", "text_keyword"] 

company_str_df = company_df[company_str_cols] 
company_numerical_df = company_df[company_numerical_cols] 


company_str_x = [] 
for i in tqdm(range(company_str_df.shape[0])): 
    cur_str = "" 
    for idx, col in enumerate(company_str_df.columns):
        cur_str += f"{col}: {company_str_df[col].values[i]}" + "[SEP]" 
    company_str_x.append(cur_str)
    
company_input_ids, company_attn_masks = [], [] 

for i in tqdm(range(len(company_str_x)), position=0, leave=True):
    encoded_inputs = tokenizer(company_str_x[i], max_length=156, truncation=True, padding="max_length")
    company_input_ids.append(encoded_inputs["input_ids"]) 
    company_attn_masks.append(encoded_inputs["attention_mask"]) 
    
company_input_ids = torch.tensor(company_input_ids, dtype=int) 
company_attn_masks = torch.tensor(company_attn_masks, dtype=int) 

print(company_input_ids.shape, company_attn_masks.shape) 

company_numerical_inputs = company_numerical_df.values 
company_numerical_inputs = torch.tensor(company_numerical_inputs).float()

recruitment_seqs = company_df["recruitment_seq"].values 

company_dict = {} 
for seq in recruitment_seqs:
    company_dict[seq] = {"input_ids":[], "attn_masks":[], "numerical_inputs":[]}
    
for i in tqdm(range(len(company_numerical_inputs)), position=0, leave=True):
    company_dict[recruitment_seqs[i]]["input_ids"] = company_input_ids[i] 
    company_dict[recruitment_seqs[i]]["attn_masks"] = company_attn_masks[i] 
    company_dict[recruitment_seqs[i]]["numerical_inputs"] = company_numerical_inputs[i] 
    
# define models 
# we need two encoders, one for candidate and one for recruitment 
class MeanPooling(nn.Module): 
    def __init__(self): 
        super(MeanPooling, self).__init__() 
    def forward(self, last_hidden_state, attention_masks): 
        input_mask_expanded = attention_masks.unsqueeze(-1).expand(last_hidden_state.size()).float() 
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) 
        sum_mask = input_mask_expanded.sum(1) 
        sum_mask = torch.clamp(sum_mask, min=1e-9) 
        mean_embeddings = sum_embeddings / sum_mask 
        return mean_embeddings  


class CandidateEmbedder(nn.Module): 
    def __init__(self):
        super(CandidateEmbedder, self).__init__() 
        self.num_embedder = nn.Linear(11, 32) 
        self.str_embedder = AutoModel.from_pretrained("monologg/kobigbird-bert-base")
        self.mean_pooler = MeanPooling() 
        merged_embedding_dim = 768 + 32 
        self.merge_layer = nn.Sequential(
            nn.Linear(merged_embedding_dim, 400), 
            nn.Mish(), 
            nn.Dropout(0.1), 
            nn.Linear(400, 200), 
            nn.Mish(), 
            nn.Dropout(0.1), 
            nn.Linear(200, 100) 
        )
    def forward(self, x_numerical, x_input_ids, x_attn_masks):
        str_embedding = self.str_embedder(x_input_ids, x_attn_masks)[0]
        str_embedding = self.mean_pooler(str_embedding, x_attn_masks)
        num_embedding = self.num_embedder(x_numerical) 
        merged_embedding = torch.cat((str_embedding, num_embedding), dim=1) 
        merged_embedding = self.merge_layer(merged_embedding) 
        return merged_embedding 

    
class CompanyEmbedder(nn.Module):
    def __init__(self):
        super(CompanyEmbedder, self).__init__() 
        self.num_embedder = nn.Linear(11, 32) 
        self.str_embedder = AutoModel.from_pretrained("monologg/kobigbird-bert-base")
        self.mean_pooler = MeanPooling() 
        merged_embedding_dim = 768 + 32 
        self.merge_layer = nn.Sequential(
            nn.Linear(merged_embedding_dim, 400), 
            nn.Mish(), 
            nn.Dropout(0.1), 
            nn.Linear(400, 200), 
            nn.Mish(), 
            nn.Dropout(0.1), 
            nn.Linear(200, 100) 
        )
    def forward(self, x_numerical, x_input_ids, x_attn_masks):
        str_embedding = self.str_embedder(x_input_ids, x_attn_masks)[0]
        str_embedding = self.mean_pooler(str_embedding, x_attn_masks)
        num_embedding = self.num_embedder(x_numerical) 
        merged_embedding = torch.cat((str_embedding, num_embedding),dim=1) 
        merged_embedding = self.merge_layer(merged_embedding) 
        return merged_embedding  
    
#학습, 검증 분리
train, val = [], []
apply_train_groupby = apply_train_df.groupby('resume_seq')['recruitment_seq'].apply(list)
for uid, iids in zip(apply_train_groupby.index.tolist(), apply_train_groupby.values.tolist()):
    for iid in iids[:-1]:
        train.append([uid,iid])
    val.append([uid, iids[-1]])

train = pd.DataFrame(train, columns=['resume_seq', 'recruitment_seq'])
val = pd.DataFrame(val, columns=['resume_seq', 'recruitment_seq'])

ids = 0 
resume_to_id = {} 
resume_seq = apply_train_df["resume_seq"].values 
for seq in resume_seq:
    if seq not in resume_to_id.keys():
        resume_to_id[seq] = ids 
        ids += 1 
        
class PairData(Dataset):
    def __init__(self, df):
        super(PairData, self).__init__() 
        self.data = df
    def __getitem__(self, index): 
        return self.data.iloc[index] 
    def __len__(self): 
        return self.data.shape[0]
    
    
class custom_collate(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
        self.chunk_size = 512 
    def __call__(self, batch): 
        q_input_ids, q_attn_masks, q_numericals, q_labels = [], [], [], []
        p_input_ids, p_attn_masks, p_numericals, p_labels = [], [], [], []  
        ids = 0 
        all_queries = [] 
        for idx, row in enumerate(batch):
            candidate, company = row[0], row[1] 
            candidate_information = candidate_dict[candidate]
            company_information = company_dict[company] 
            
            q_input_ids.append(candidate_information["input_ids"]) 
            q_attn_masks.append(candidate_information["attn_masks"]) 
            q_numericals.append(candidate_information["numerical_inputs"]) 
            q_labels.append(resume_to_id[candidate])
            
            p_input_ids.append(company_information["input_ids"]) 
            p_attn_masks.append(company_information["attn_masks"])
            p_numericals.append(company_information["numerical_inputs"]) 
            p_labels.append(resume_to_id[candidate])  
        
        q_input_ids = torch.stack(q_input_ids, dim=0).squeeze(dim=1) 
        q_attn_masks = torch.stack(q_attn_masks, dim=0).squeeze(dim=1) 
        q_numericals = torch.stack(q_numericals, dim=0).squeeze(dim=1) 
        q_labels = torch.tensor(q_labels)
        
        p_input_ids = torch.stack(p_input_ids, dim=0).squeeze(dim=1) 
        p_attn_masks = torch.stack(p_attn_masks, dim=0).squeeze(dim=1) 
        p_numericals = torch.stack(p_numericals, dim=0).squeeze(dim=1) 
        p_labels = torch.tensor(p_labels)
        
        return q_input_ids, q_attn_masks, q_numericals, q_labels, p_input_ids, p_attn_masks, p_numericals, p_labels 
    
    
class custom_collate_inference(object): 
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base") 
        self.chunk_size = 512 
    def __call__(self, batch): 
        p_input_ids, p_attn_masks, p_numericals = [], [], [] 
        for idx, row in enumerate(batch):
            candidate, company = row[0], row[1] 
            candidate_information = candidate_dict[candidate]
            company_information = company_dict[company] 
            
            p_input_ids.append(company_information["input_ids"]) 
            p_attn_masks.append(company_information["attn_masks"])
            p_numericals.append(company_information["numerical_inputs"]) 
        p_input_ids = torch.stack(p_input_ids, dim=0).squeeze(dim=1) 
        p_attn_masks = torch.stack(p_attn_masks, dim=0).squeeze(dim=1) 
        p_numericals = torch.stack(p_numericals, dim=0).squeeze(dim=1) 
        return p_input_ids, p_attn_masks, p_numericals
    
device = torch.device("cuda") 

collate = custom_collate() 
train_set = PairData(train)
train_dataloader = DataLoader(train_set, batch_size=32, collate_fn=collate, shuffle=True) 

val_set = PairData(val) 
val_dataloader = DataLoader(val_set, batch_size=32, collate_fn=collate, shuffle=False)

collate_inference = custom_collate_inference() 
full_set = PairData(apply_train_df) 
full_dataloader = DataLoader(full_set, batch_size=32, collate_fn=collate_inference, shuffle=False)

candidate_encoder = CandidateEmbedder()
candidate_encoder.to(device)

company_encoder = CompanyEmbedder() 
company_encoder.to(device) 

query_index_dict = {} 

resume_seq = apply_train_df["resume_seq"].values 

for seq in resume_seq: 
    query_index_dict[seq] = []

for i in range(len(resume_seq)): 
    query_index_dict[resume_seq[i]].append(i)

## training logic 
epochs = 10 
params = list(candidate_encoder.parameters()) + list(company_encoder.parameters()) 
optimizer = torch.optim.AdamW(params, lr=2e-5, eps=1e-8) 
t_total = len(train_dataloader) * epochs 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*t_total), num_training_steps=t_total) 
miner = miners.MultiSimilarityMiner() 
loss_func = losses.MultiSimilarityLoss() 

candidate_encoder.zero_grad() 
company_encoder.zero_grad() 
torch.cuda.empty_cache() 

best_recall = 0

train_losses , val_losses = [], [] 
val_recall_5 = [] 

for epoch in tqdm(range(0, epochs), desc="Epochs", position=0, leave=True, total=epochs):
    train_loss = 0 
    candidate_encoder.train() 
    company_encoder.train() 
    with tqdm(train_dataloader, unit="batch") as tepoch: 
        for step, batch in enumerate(tepoch): 
            batch = tuple(t.to(device) for t in batch) 
            q_input_ids, q_attn_masks, q_numericals, q_labels, p_input_ids, p_attn_masks, p_numericals, p_labels = batch 
            q_embeddings = candidate_encoder(q_numericals, q_input_ids, q_attn_masks) 
            p_embeddings = company_encoder(p_numericals, p_input_ids, p_attn_masks) 
            full_embeddings = torch.cat((q_embeddings, p_embeddings), dim=0)
            full_labels = torch.cat((q_labels, p_labels))
            hard_pairs = miner(full_embeddings, full_labels) 
            loss = loss_func(full_embeddings, full_labels, hard_pairs) 
            train_loss += loss.item() 
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step() 
            scheduler.step() 
            candidate_encoder.zero_grad() 
            company_encoder.zero_grad() 
            tepoch.set_postfix(loss=train_loss / (step+1))
            time.sleep(0.1) 
    avg_train_loss = train_loss / len(train_dataloader) 
    
    val_loss = 0 
    candidate_encoder.eval() 
    company_encoder.eval() 
    for step, batch in tqdm(enumerate(val_dataloader), position=0, leave=True, total=len(val_dataloader)):
        batch = tuple(t.to(device) for t in batch) 
        q_input_ids, q_attn_masks, q_numericals, q_labels, p_input_ids, p_attn_masks, p_numericals, p_labels = batch 
        with torch.no_grad(): 
            q_embeddings = candidate_encoder(q_numericals, q_input_ids, q_attn_masks) 
            p_embeddings = company_encoder(p_numericals, p_input_ids, p_attn_masks) 
            full_embeddings = torch.cat((q_embeddings, p_embeddings), dim=0) 
            full_labels = torch.cat((q_labels, p_labels), dim=0) 
            loss = loss_func(full_embeddings, full_labels) 
            val_loss += loss.item() 
    avg_val_loss = val_loss / len(val_dataloader) 
    train_losses.append(avg_train_loss) 
    val_losses.append(avg_val_loss) 
    
    # calculate recall
    with torch.no_grad(): 
        recall = 0
        candidate_encoder.eval() 
        company_encoder.eval() 
        company_embs = [] 
        for step, batch in tqdm(enumerate(full_dataloader), position=0, leave=True, total=len(full_dataloader)): 
            batch = (t.to(device) for t in batch) 
            p_input_ids, p_attn_masks, p_numericals = batch 
            p_embs = company_encoder(p_numericals, p_input_ids, p_attn_masks) 
            p_embs = p_embs.detach().cpu() 
            for i in range(p_embs.shape[0]): 
                company_embs.append(torch.reshape(p_embs[i], (-1, 100)))
        
        company_embs = torch.cat(company_embs, dim=0) 
        company_embs = company_embs.detach().cpu().numpy() 
        print(f"candidate embeddings shape: {company_embs.shape}") 
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(100))
        company_embs = company_embs.astype(np.float32)
        faiss.normalize_L2(company_embs)
        index.add_with_ids(company_embs, np.array(range(0, len(company_embs)), dtype=int))
        index.nprobe = 64
        
        top_5 = 0 
        val_seqs = val["resume_seq"].values 
        for sample_idx in tqdm(range(len(val_seqs)), position=0, leave=True): 
            query = val_seqs[sample_idx] 
            b_input_ids = candidate_dict[query]["input_ids"].to(device) 
            b_input_ids = torch.reshape(b_input_ids, (1, -1)) 
            
            b_attn_masks = candidate_dict[query]["attn_masks"].to(device) 
            b_attn_masks = torch.reshape(b_attn_masks, (1, -1))
            
            b_numericals = candidate_dict[query]["numerical_inputs"].to(device)
            b_numericals = torch.reshape(b_numericals, (1, -1))
            
            candidate_embedding = company_encoder(b_numericals, b_input_ids, b_attn_masks) 
            candidate_embedding = candidate_embedding.detach().cpu().numpy() 
            distances, indices = index.search(candidate_embedding, 1000)
            correct_idx = query_index_dict[query] 
            
            cnt = 0 
            for idx in correct_idx: 
                if idx in indices[0][:5]:
                    cnt += 1 
            top_5 += cnt / min(len(correct_idx), 5) 
            
        avg_top_5 = top_5 / len(val_seqs)
        
        if avg_top_5 > best_recall: 
            best_recall = avg_top_5 
            torch.save(candidate_encoder.state_dict(), f"candidate_encoder.pt") 
            torch.save(company_encoder.state_dict(), f"company_encoder.pt") 
            print("saved current best checkpoint!") 
            
    print(f"Epoch: {epoch} | avg train loss: {avg_train_loss} | avg val loss: {avg_val_loss} | recall: {avg_top_5}") 
    
