import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from email.policy import strict
from utils.utils import *
from utils.sinkhorn_knopp import *
from model import *
from dataloader import *
import warnings
# import ot
import math
import time
import json
from argparse import ArgumentParser
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from sklearn import mixture
from transformers import logging, WEIGHTS_NAME
from init_parameter import init_model
from pretrain import PretrainModelManager
from model import BertForOT, BertForModel
import seaborn as sn
class Manager:

    def __init__(self, args, data, pretrained_model):
        set_seed(args.seed)
        args.method  = 'bias'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pretrained_model is None:
            pretrained_model = BertForModel(args.bert_model, num_labels=data.n_known_cls)
            if os.path.exists(args.pretrain_dir):
                model_file = os.path.join(args.pretrain_dir, 'premodel.pth')
                pretrained_model.load_state_dict(torch.load(model_file))
        self.pretrained_model = pretrained_model.to(self.device)
        if args.cluster_num_factor > 1:
            data.num_labels = self.predict_k(args, data) 
        print(data.num_labels)
        self.model = BertForOT(args.bert_model, num_labels=data.num_labels)
        self.model.to(self.device)
        self.load_pretrained_model()
        # self.evaluation(data)
        self.initialize_classifier(args, data)
        self.freeze_parameters(self.model)
        self.num_train_optimization_steps = int(len(data.train_labeled_examples) / args.train_batch_size) * args.num_pretrain_epochs
        self.optimizer, self.scheduler = self.get_optimizer(args)

        
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)
        self.sinkhorn = SinkhornKnopp(args)

    def train(self, args, data):

        unlabeled_iter = iter(data.train_semi_dataloader)

        for epoch in range(int(args.num_train_epochs)):

            print('---------------------------')
            print(f'training epoch:{epoch}')

            self.model.train()

            if epoch == 0:
                optimal_map, _ = self.ot_kt(data)

            factor = self.exponential_decay(epoch, 0.4, 80, 0.3)
            threshold = self.exponential_decay(epoch, 0.3, 80, 0.4)

            for batch in tqdm(data.train_labeled_dataloader, desc="Pseudo-label training"):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, labels = batch
                X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                # a weird bug to reproduce the results for banking
                if args.dataset == 'banking':
                    X_l2 = {"input_ids": self.generator.random_token_replace(input_ids.cpu()).to(self.device), "attention_mask": input_mask, "token_type_ids": segment_ids}
                labels_hard = labels
                labels = torch.zeros(len(labels), data.num_labels, device=self.device).scatter_(1, labels.view(-1,1).long(), 1)

                try:
                    batch_u = unlabeled_iter.next()
                    batch_u = tuple(t.to(self.device) for t in batch_u)
                except StopIteration:
                    unlabeled_iter = iter(data.train_semi_dataloader)
                    batch_u = unlabeled_iter.next()
                    batch_u = tuple(t.to(self.device) for t in batch_u)
                input_ids, input_mask, segment_ids, labels_u = batch_u
                X_u1 = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                X_u2 = {"input_ids": self.generator.random_token_replace(input_ids.cpu()).to(self.device), "attention_mask": input_mask, "token_type_ids": segment_ids}
                
                with torch.no_grad():
                    _, logits_bias = self.pretrained_model(X_u1, output_hidden_states=True)
                    probs = torch.softmax(logits_bias, dim=1)
                    entropy = torch.sum(-probs*torch.log(probs), dim=1)

                with torch.no_grad():
                    w = self.model.classifier.weight.data.clone()
                    w = F.normalize(w, dim=1, p=2)
                    self.model.classifier.weight.copy_(w)

                with torch.no_grad():
                    # logits adjustment
                    _, logits_u1 = self.model(X_u1)
                    self.logits_adjustment(data, logits_u1, logits_bias, optimal_map, entropy)
                    _, logits_u2 = self.model(X_u2)
                    self.logits_adjustment(data, logits_u2, logits_bias, optimal_map, entropy)

                    labels_u1 = self.sinkhorn(logits_u2)
                    labels_u2 = self.sinkhorn(logits_u1)


                hard_novel_idx1 = self.split_hard_novel_soft_seen(data, labels_u1, threshold)
                hard_novel_idx2 = self.split_hard_novel_soft_seen(data, labels_u2, threshold)


                self.gen_hard_novel(labels_u1, hard_novel_idx1, threshold)
                self.gen_hard_novel(labels_u2, hard_novel_idx2, threshold)
 


                X_u = {"input_ids": torch.cat([X_u1["input_ids"], X_u2["input_ids"]], dim=0), 
                       "attention_mask": torch.cat([X_u1["attention_mask"], X_u2["attention_mask"]], dim=0),
                       "token_type_ids": torch.cat([X_u1["token_type_ids"], X_u2["token_type_ids"]], dim=0)}
                labels_u = torch.cat([labels_u1, labels_u2], dim=0)

                feats_l, logits_l = self.model(X)
                feats_l = F.normalize(feats_l, dim=1)
                if args.dataset == 'banking':
                    feats_l2, logits_l2 = self.model(X_l2)
                    feats_l2 = F.normalize(feats_l2, dim=1)
                    contrastive_feats = torch.cat((feats_l.unsqueeze(1), feats_l2.unsqueeze(1)), dim = 1)

                feats_u, logits_u = self.model(X_u)
                


                feats_u1 = feats_u[:len(labels_u1), :]
                feats_u2 = feats_u[len(labels_u1):, :]
                

                logits_l = F.normalize(logits_l, dim=1)
                logits_u = F.normalize(logits_u, dim=1)
                
                
                

                feats_u1 = F.normalize(feats_u1, dim=1)
                feats_u2 = F.normalize(feats_u2, dim=1)

                # a weird bug to reproduce the results for banking


                loss_cel = -torch.mean(torch.sum(labels * F.log_softmax((logits_l/0.3), dim=1), dim=1))
                loss_ceu = -torch.mean(torch.sum(labels_u * F.log_softmax((logits_u/0.3), dim=1), dim=1))



                loss_contrast = self.model.loss_contrast(feats_u1, feats_u2, 0.07)
                loss_scl = self.model.loss_scl(feats_l.unsqueeze(1), labels_hard)
                
                loss = (1-factor) * loss_ceu + factor * loss_cel  + 0.02 * loss_scl + 0.01 * loss_contrast

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            

    def initialize_classifier(self, args, data):
        # extract labeled prototypes
        feats, labels = self.get_features_labels(data.train_labeled_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        [rows, _] = feats.shape
        num = np.zeros(data.n_known_cls)
        self.proto_l = np.zeros((data.n_known_cls, args.feat_dim))
        for i in range(rows):
            self.proto_l[labels[i]] += feats[i]
            num[labels[i]] += 1
        for i in range(data.n_known_cls):
            self.proto_l[i] = self.proto_l[i] / num[i]

        # extract and align unlabeled prototypes
        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels, n_init=20).fit(feats)
        self.proto_u = km.cluster_centers_
        distance = dist.cdist(self.proto_l, self.proto_u, 'euclidean')
        _, col_ind = linear_sum_assignment(distance)
        pro_l = []
        for i in range(len(col_ind)):
            pro_l.append(self.proto_u[col_ind[i]][:])
        pro_u = []
        for j in range(data.num_labels):
            if j not in col_ind:
                pro_u.append(self.proto_u[j][:])
        self.proto_u = pro_l + pro_u   
        self.proto_u = torch.tensor(np.array(self.proto_u), dtype=torch.float).to(self.device)

        # initialize prototype classifier
        self.model.classifier.weight.data = self.proto_u
        with torch.no_grad():
            w = self.model.classifier.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.model.classifier.weight.copy_(w)

    def ot_kt(self, data):
    
        pro_u = self.proto_u[len(self.proto_l):].cpu()
        optimal_map = F.normalize(torch.tensor(self.proto_l, dtype=torch.float), dim=1) @ F.normalize(torch.tensor(pro_u, dtype=torch.float).t(), dim=0)
        optimal_map = F.softmax(optimal_map)
        optimal_map = optimal_map.to(self.device)

        padding = torch.zeros(data.n_known_cls, data.n_known_cls, dtype=torch.float).to(self.device)
        optimal_map_padding = torch.cat((padding, optimal_map), 1)

        return optimal_map, optimal_map_padding


    def logits_adjustment(self, data, logits, logits_bias, optimal_map, entropy):
        mask =  torch.sigmoid(entropy - torch.max(entropy))
        mask = mask.reshape(-1, 1)
        logits[:, :data.n_known_cls] -= mask * logits_bias * data.beta
        logits[:, data.n_known_cls:] += mask * (logits_bias @ optimal_map) * data.beta


    def split_hard_novel_soft_seen(self, data, labels, threshold):
        labels_novel = labels[:, data.n_known_cls:]
        max_pred_novel, _ = torch.max(labels_novel, dim=-1)
        hard_novel_idx = torch.where(max_pred_novel>=threshold)[0]


        return hard_novel_idx

    def gen_hard_novel(self, labels, hard_novel_idx, threshold):
        labels[hard_novel_idx] = labels[hard_novel_idx].ge(threshold).float()

    def get_optimizer(self, args):
        num_warmup_steps = int(args.warmup_proportion * self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        return optimizer, scheduler

    def freeze_parameters(self, model):
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
            if "encoder.layer.8" in name or "encoder.layer.9" in name or "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def get_features_labels(self, dataloader, model, args):
        model.eval()
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation for clustering"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature, _ = model(X, output_hidden_states=True)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def load_pretrained_model(self):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def evaluation(self, data):
        self.model.eval()
        pred_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        feats = torch.empty((0, 768)).to(self.device)

        for batch in data.test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feat, logits = self.model(X, output_hidden_states=True)
            labels = torch.argmax(logits, dim=1)

            pred_labels = torch.cat((pred_labels, labels))  
            total_labels = torch.cat((total_labels, label_ids))  
            feats = torch.cat((feats, feat))

        y_pred = pred_labels.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        results = clustering_score(y_true, y_pred, data.known_lab)
        print('results', results)

        km = KMeans(n_clusters = data.num_labels, n_init=20).fit(feats.cpu().numpy())
        y_pred1 = km.labels_
        results1 = clustering_score(y_true, y_pred1, data.known_lab)
        print('results1', results1)

    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def exponential_decay(self, t, init, m, finish):
        alpha = np.log(init / finish) / m
        l = - np.log(init) / alpha
        decay = np.exp(-alpha * (t + l))
        return decay

    def predict_k(self, args, data):
        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.pretrained_model.cuda(), args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels * 0.9
        print('drop',drop_out)

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        num_labels = len(pred_label_list) - cnt
        print('pred_num',num_labels)

        return num_labels

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    if args.pretrain:
        print('Pre-training begin...')
        manager_p = PretrainModelManager(args, data)
        manager_p.train(args, data)
        print('Pre-training finished!')
        del manager_p
    manager_p = PretrainModelManager(args, data)
    manager_p.load_model(args)
    manager = Manager(args, data, manager_p.model)
    
    manager.evaluation(data)
    print('Training begin...')
    manager.train(args,data)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(data)
    print('Evaluation finished!')