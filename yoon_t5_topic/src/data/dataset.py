from torch.utils.data import Dataset
import torch

# Train에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForTrain(Dataset):
    """T5용 간소화된 Dataset 클래스"""
    def __init__(self, encoder_input, labels, length, pad_token_id):
        self.encoder_input = encoder_input
        self.labels = labels
        self.length = length
        self.pad_token_id = pad_token_id

    def __getitem__(self, idx):
        labels = self.labels['input_ids'][idx].clone().to(torch.long)
        # ✅ tokenizer.pad_token_id를 확실히 사용
        labels[labels == self.pad_token_id] = -100
        
        item = {
            'input_ids': self.encoder_input['input_ids'][idx].clone().detach().to(torch.long),
            'attention_mask': self.encoder_input['attention_mask'][idx].clone().detach().to(torch.long),
            'labels': labels
        }
        return item

    def __len__(self):
        return self.length

# Validation에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForVal(Dataset):
    """T5용 간소화된 Dataset 클래스"""
    def __init__(self, encoder_input, labels, length):
        self.encoder_input = encoder_input
        self.labels = labels
        self.length = length

    def __getitem__(self, idx):
        labels = self.labels['input_ids'][idx].clone().to(torch.long)
        # ✅ 검증도 동일하게 pad → -100
        labels[labels == self.pad_token_id] = -100

        item = {
            'input_ids': self.encoder_input['input_ids'][idx].clone().detach().to(torch.long),
            'attention_mask': self.encoder_input['attention_mask'][idx].clone().detach().to(torch.long),
            'labels': labels
        }
        return item

    def __len__(self):
        return self.length

# Test에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForInference(Dataset):
    """추론용 Dataset"""
    def __init__(self, encoder_input, test_id, length):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.length = length

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encoder_input['input_ids'][idx].clone().detach(),
            'attention_mask': self.encoder_input['attention_mask'][idx].clone().detach(),
            'ID': self.test_id[idx]
        }
        return item

    def __len__(self):
        return self.length
