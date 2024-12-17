import torch
from torch import nn
from torch.nn.parameter import Parameter
from utils.contrastive import SupConLoss
from utils.utils import *

class BertForModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, self.num_labels)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        outputs = self.backbone(**X, output_hidden_states=True)
        CLSEmbedding = torch.mean(outputs.hidden_states[-1][:, 1:], dim=1)
        CLSEmbedding = self.dropout(CLSEmbedding)

        logits = self.classifier(CLSEmbedding)

        return CLSEmbedding, logits

    def mlmForward(self, X, Y):
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, y)
        return output 

    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)

    def loss_scl(self, embds, label=None, mask=None, temperature=0.07, base_temperature=0.07):
        """compute contrastive loss"""
        loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        output = loss(embds, labels=label, mask=mask)
        return output    

class BertForOT(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForOT, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, self.num_labels, bias=False)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        outputs = self.backbone(**X, output_hidden_states=True)
        CLSEmbedding = torch.mean(outputs.hidden_states[-1][:, 1:], dim=1)
        CLSEmbedding = self.dropout(CLSEmbedding)

        logits = self.classifier(F.normalize(CLSEmbedding, dim=1))

        return CLSEmbedding, logits

    def loss_contrast(self, emb_i, emb_j, temperature):		
        z_i = F.normalize(emb_i, dim=1)    
        z_j = F.normalize(emb_j, dim=1)    

        representations = torch.cat([z_i, z_j], dim=0)          
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      
        
        sim_ij = torch.diag(similarity_matrix, len(emb_i))         
        sim_ji = torch.diag(similarity_matrix, -len(emb_i))        
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  
        negatives_mask = (~torch.eye(len(emb_i) * 2, len(emb_i) * 2, dtype=bool).to(emb_i.device)).float()

        nominator = torch.exp(positives / temperature)             
        denominator = negatives_mask * torch.exp(similarity_matrix / temperature)             
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))       
        loss = torch.sum(loss_partial) / (2 * len(emb_i))
        return loss
    
    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)
    
    def loss_scl(self, embds, label=None, mask=None, temperature=0.07, base_temperature=0.07):
        """compute contrastive loss"""
        loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        output = loss(embds, labels=label, mask=mask)
        return output