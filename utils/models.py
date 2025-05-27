import torch.nn as nn
import torch
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

class BERTGroupClassifier(nn.Module):
    def __init__(self, model_name, num_labels=1):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get embeddings from the model
        batch_size, max_k, max_seq_len = input_ids.shape
        input_ids = input_ids.view(-1, max_seq_len)  # Flatten to (batch_size * max_k, max_seq_length)
        attention_mask = attention_mask.view(-1, max_seq_len)  # Same flattening

        # Step 1: Pass through BERT (embed all the messages)
        # Shape: (batch_size * max_k * max_seq_len, hidden_size)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs_reshaped = outputs.last_hidden_state.reshape(batch_size, max_k, max_seq_len, -1)
        
        # Step 2: Slice the [CLS] token (at index 0 in seq_len dimension) for each chunk
        # Shape: (batch_size, max_k, hidden_size)
        cls_token_states = outputs_reshaped[:, :, 0, :]  

        # Step 3: Max Pool embeddings across chunks (now with temporal information)
        # Shape: (batch_size, hidden_size)
        pooled_output_max = torch.max(cls_token_states, dim=1).values  

        # Step 4: Apply classification head
        # Shape: (batch_size, num_labels)
        logits = self.fc(pooled_output_max)              
        loss = None

        if labels is not None:
            labels = labels.float().view(-1, 1)  # Ensure labels are float for BCE loss
            loss_fct = nn.BCEWithLogitsLoss()  # Automatically applies sigmoid
            loss = loss_fct(logits, labels)  # Ensure correct shape
           # Always return (loss, logits) for consistency
        return SequenceClassifierOutput(loss=loss,logits=logits)
