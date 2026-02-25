import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import TokenEmbedding
from modules import MLA,LayerNorm,MoE,TokenEmbedding

class ModelBlock(nn.Module):
    def __init__(
            self,d_model,n_head,d_c,d_r,device,hidden,
            other_experts,shared_experts,keep,
            dropout=0.1,ro_theta=10000.0,scale=0.02
        ):
        super().__init__()
        self.mla = MLA(d_model,d_c,d_r,n_head,device,ro_theta)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.moe_ffn = MoE(other_experts,shared_experts,d_model,hidden,device,keep,scale)
        self.drop = nn.Dropout(dropout)
    
    def forward(self,x,mask=None):
        attout = self.mla(x,x,mask)
        x = self.norm1(x+self.drop(attout))

        ffnout,lose = self.moe_ffn(x)
        x = self.norm2(x+self.drop(ffnout))
        return x, lose
    
class shared_TinySeek(nn.Module):
    def __init__(self,config,NO_loss_p=True):
        super().__init__()
        self.tokenEmbedding = TokenEmbedding(config['d_model'],config['vocab_size'])
        self.n_layers = config['n_layer']
        self.synonymous_layer = ModelBlock(
            config['d_model'],config['n_head'],config['d_c'],config['d_r'],
            config['device'],config['hidden'],
            config['other_experts'],config['shared_experts'],
            config['keep'],
            config['dropout'],config['ro_theta'],config['scale']
            )
        self.pad_idx = config['pad_idx']
        self.fc = nn.Linear(config['d_model'],config['vocab_size'],bias=True)
        # self.fc.weight = self.tokenEmbedding.tokenEmbedding.weight
        self.device = config['device']
        # self.fc_scale = nn.Parameter(torch.ones(1,1,1).to(self.device))*config['scale']
        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']
        self.NO_loss_p = NO_loss_p
    
    def get_mask(self, seq):
        _, seq_len = seq.shape
        
        # 计算每个样本的有效长度（非填充部分的长度）
        valid = (seq != self.pad_idx).bool()  # [batch_size]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        pad_mask = (seq != self.pad_idx).unsqueeze(1).unsqueeze(3)  # shape: [batch_size, 1, seq_len, 1]
        pad_mask = pad_mask.repeat(1, 1, 1, seq_len)  # [batch_size, 1, seq_len, seq_len]
        
        combined_mask = causal_mask & pad_mask
        return combined_mask, valid
    
    def forward(self, x, labels=None, EVAL=False, n_layers=6):
        batch,_ = x.shape

        mask,valid = self.get_mask(x)
        x = self.tokenEmbedding(x)
        if labels is None:
            EVAL = True
        if EVAL is False:
            embedding_labels = torch.cat([x[:, 1:, :], torch.zeros_like(x[:, :1, :])], dim=1)
            embedding_labels[-1] = self.tokenEmbedding.tokenEmbedding.weight[0]

            total_loss = 0.0
            total_distance = 0.0

            if self.NO_loss_p is True:
                with torch.no_grad():
                    for _ in range(self.n_layers):
                        x, loss = self.synonymous_layer(x,mask)
                        distance = F.mse_loss(x[valid],embedding_labels[valid])
                        total_distance += distance
                        total_loss += loss
            else:
                for _ in range(self.n_layers):
                    x, loss = self.synonymous_layer(x,mask)
                    distance = F.mse_loss(x[valid],embedding_labels[valid])
                    total_distance += distance
                    total_loss += loss
            
            # E_norm = F.normalize(self.tokenEmbedding.tokenEmbedding.weight,dim=-1)
            # gram = E_norm.T @ E_norm  # (d_model, d_model)
            # identity = torch.eye(self.d_model, device=gram.device)
            # orthogonal_constraint = F.mse_loss(gram, identity)

            output = self.fc(x) #*self.fc_scale
            cross_loss = F.cross_entropy(input=output.reshape(-1,self.vocab_size),
                                    target=labels.reshape(-1),
                                    ignore_index = self.pad_idx)
            return total_distance,cross_loss,total_loss

        else:
            for index in range(self.n_layers):
                new_x,_ = self.synonymous_layer(x,mask)
                # distance = F.mse_loss(x[valid],new_x[valid])
                x = new_x
                # if distance<0.0005:
                #    print(f"\n思考{index+1}层:",end=' ')
                #    break
            output = self.fc(x) #*self.fc_scale
            output_2 = torch.ones_like(output)
            
            return output,output_2,0