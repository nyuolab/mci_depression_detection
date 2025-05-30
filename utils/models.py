import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import get_peft_model, LoraConfig, TaskType

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([1 - alpha, alpha])  # [weight for class 0, class 1]
        elif isinstance(alpha, list) or isinstance(alpha, torch.Tensor):
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits of shape (batch_size, num_classes)
        targets: ground-truth labels of shape (batch_size,) with class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # shape (batch_size,)
        pt = torch.exp(-ce_loss)  # pt = probability of the true class
        at = self.alpha.to(inputs.device)[targets]  # alpha for each example
        loss = at * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class BERTGroupClassifier(nn.Module):
    def __init__(self, model_name, num_labels=1):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.gradient_checkpointing_enable()
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
            labels = labels.long().view(-1)  # Ensure labels are float for BCE loss
            loss_fct = FocalLoss(alpha=0.95, gamma=2.0)
            loss = loss_fct(logits, labels)  # Ensure correct shape
           # Always return (loss, logits) for consistency
        return SequenceClassifierOutput(loss=loss,logits=logits)

#Not ready yet
class DecoderClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, lora_r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1):
        super().__init__()

        base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=False,
            output_hidden_states=True
        )

        # Setup PEFT LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,   # sequence classification
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj"]  # attention projections, adjust if needed
        )

        # Wrap base model with PEFT LoRA
        self.base_model = get_peft_model(base_model, peft_config)
        hidden_size = self.base_model.config.hidden_size

        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.loss_fn = FocalLoss(alpha=0.95, gamma=2.0)
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)

        # Mean pooling over attention mask
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1)

        logits = self.classifier(pooled)
        if labels is not None:
            labels = labels.long()  
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}



