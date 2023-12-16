# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional, Tuple
from transformers import BertTokenizer, BertForSequenceClassification, LongformerSelfAttention
import GPUtil
import math


class LongBertSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.long_self_attn = LongformerSelfAttention(config, layer_id=layer_id)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):

        batch_size, seq_len, _ = hidden_states.size()
        attention_mask = attention_mask.view(batch_size, seq_len)

        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        outputs = self.long_self_attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=None,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )

        return outputs

class LongBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.bert.encoder.layer):
            layer.attention.self = LongBertSelfAttention(config, layer_id=i)
    

def expand_bert_token_size(model, tokenizer, max_token):
    config = model.config
    embeddings = model.bert.embeddings
    cur_max_token = embeddings.position_embeddings.weight.size(0)
    assert max_token > cur_max_token
    
    num_repeats = math.ceil(max_token / cur_max_token)
    
    config.max_position_embeddings = max_token
    embed_weight = embeddings.position_embeddings.weight.data
    embed_weight = embed_weight.repeat(num_repeats, 1)[:max_token,:]
    embeddings.position_embeddings.weight.data = embed_weight
    
    token_type_ids = embeddings.token_type_ids
    token_type_ids = token_type_ids.repeat(1, num_repeats)[:,:max_token]
    embeddings.token_type_ids = token_type_ids
    
    embeddings.position_ids.data = torch.arange(0, max_token).view(1, -1)
    tokenizer.model_max_length = max_token
    tokenizer.init_kwargs['max_len'] = max_token
    
def convert_bert_with_longformer(model, tokenizer, max_token, attn_win):
    expand_bert_token_size(model, tokenizer, max_token)
    
    config = model.config
    config.attention_window = [attn_win] * config.num_hidden_layers
    for i, layer in enumerate(model.bert.encoder.layer):
        longformer_self_attn = LongBertSelfAttention(config, layer_id=i)
        longformer_self_attn.long_self_attn.query = layer.attention.self.query
        longformer_self_attn.long_self_attn.key = layer.attention.self.key
        longformer_self_attn.long_self_attn.value = layer.attention.self.value
        longformer_self_attn.long_self_attn.query_global = deepcopy(layer.attention.self.query)
        longformer_self_attn.long_self_attn.key_global = deepcopy(layer.attention.self.key)
        longformer_self_attn.long_self_attn.value_global = deepcopy(layer.attention.self.value)
        layer.attention.self = longformer_self_attn


##############################################################
########################## TEST ##############################
##############################################################

text_content = "1970년대 히트곡 '나를 두고 아리랑'을 만들고 부른 작곡가 겸 가수 김중신이 지난 2일(이하 현지시간) 미국 하와이에서 지병으로 별세했다고 박성서 대중음악평론가가 8일 전했다. 향년 81세. 1942년 대구에서 태어난 고인은 대구의 호텔 나이트클럽에서 기타리스트로 음악 활동을 시작해 1971년 그룹사운드 '윤항기와 키브라더스'에서 활동했다. 1974년에는 그룹사운드 '김훈과 트리퍼스'의 '나를 두고 아리랑'을 작사·작곡했으며, 이듬해 이 노래를 현혜미와 직접 듀엣으로 불러 발표했다. '나를 두고 아리랑'은 '나를 나를 나를 두고/ 물건너 가시더니/ 한 달 두 달 해가 또 가도/ 편지 한 장 없네'라는 서글픈 가사로 인기를 끌어 고인의 대표곡으로 남았다. 박성서 대중음악평론가는 '이 곡은 우리나라 고유의 선율을 당시 유행하던 고고리듬으로 편곡하고 그룹사운드 반주를 붙여 만든 파격적인 노래'라며 '김훈과 트리퍼스는 '나를 두고 아리랑'으로 당시 10대 가수상까지 받았다'고 말했다. 고인과 함께 음악 활동을 한 윤항기는 '1970년대 당시 많은 가수가 팝적인 음악을 추구할 때, 김중신은 한국 전통 선율을 활용한 '나를 두고 아리랑'으로 우리 가요에 변화를 줬다'고 평했다. '나를 두고 아리랑'은 이후 윤항기를 비롯해 나훈아, 조미미, 이용복, 선우성 등 많은 가수가 리메이크한 1970년대 가요계 명곡 가운데 하나로 남았다. 1970년대 히트곡 '나를 두고 아리랑'을 만들고 부른 작곡가 겸 가수 김중신이 지난 2일(이하 현지시간) 미국 하와이에서 지병으로 별세했다고 박성서 대중음악평론가가 8일 전했다. 향년 81세. 1942년 대구에서 태어난 고인은 대구의 호텔 나이트클럽에서 기타리스트로 음악 활동을 시작해 1971년 그룹사운드 '윤항기와 키브라더스'에서 활동했다. 1974년에는 그룹사운드 '김훈과 트리퍼스'의 '나를 두고 아리랑'을 작사·작곡했으며, 이듬해 이 노래를 현혜미와 직접 듀엣으로 불러 발표했다. '나를 두고 아리랑'은 '나를 나를 나를 두고/ 물건너 가시더니/ 한 달 두 달 해가 또 가도/ 편지 한 장 없네'라는 서글픈 가사로 인기를 끌어 고인의 대표곡으로 남았다. 박성서 대중음악평론가는 '이 곡은 우리나라 고유의 선율을 당시 유행하던 고고리듬으로 편곡하고 그룹사운드 반주를 붙여 만든 파격적인 노래'라며 '김훈과 트리퍼스는 '나를 두고 아리랑'으로 당시 10대 가수상까지 받았다'고 말했다. 고인과 함께 음악 활동을 한 윤항기는 '1970년대 당시 많은 가수가 팝적인 음악을 추구할 때, 김중신은 한국 전통 선율을 활용한 '나를 두고 아리랑'으로 우리 가요에 변화를 줬다'고 평했다. '나를 두고 아리랑'은 이후 윤항기를 비롯해 나훈아, 조미미, 이용복, 선우성 등 많은 가수가 리메이크한 1970년대 가요계 명곡 가운데 하나로 남았다. 1970년대 히트곡 '나를 두고 아리랑'을 만들고 부른 작곡가 겸 가수 김중신이 지난 2일(이하 현지시간) 미국 하와이에서 지병으로 별세했다고 박성서 대중음악평론가가 8일 전했다. 향년 81세. 1942년 대구에서 태어난 고인은 대구의 호텔 나이트클럽에서 기타리스트로 음악 활동을 시작해 1971년 그룹사운드 '윤항기와 키브라더스'에서 활동했다. 1974년에는 그룹사운드 '김훈과 트리퍼스'의 '나를 두고 아리랑'을 작사·작곡했으며, 이듬해 이 노래를 현혜미와 직접 듀엣으로 불러 발표했다. '나를 두고 아리랑'은 '나를 나를 나를 두고/ 물건너 가시더니/ 한 달 두 달 해가 또 가도/ 편지 한 장 없네'라는 서글픈 가사로 인기를 끌어 고인의 대표곡으로 남았다. 박성서 대중음악평론가는 '이 곡은 우리나라 고유의 선율을 당시 유행하던 고고리듬으로 편곡하고 그룹사운드 반주를 붙여 만든 파격적인 노래'라며 '김훈과 트리퍼스는 '나를 두고 아리랑'으로 당시 10대 가수상까지 받았다'고 말했다. 고인과 함께 음악 활동을 한 윤항기는 '1970년대 당시 많은 가수가 팝적인 음악을 추구할 때, 김중신은 한국 전통 선율을 활용한 '나를 두고 아리랑'으로 우리 가요에 변화를 줬다'고 평했다. '나를 두고 아리랑'은 이후 윤항기를 비롯해 나훈아, 조미미, 이용복, 선우성 등 많은 가수가 리메이크한 1970년대 가요계 명곡 가운데 하나로 남았다. 1970년대 히트곡 '나를 두고 아리랑'을 만들고 부른 작곡가 겸 가수 김중신이 지난 2일(이하 현지시간) 미국 하와이에서 지병으로 별세했다고 박성서 대중음악평론가가 8일 전했다. 향년 81세. 1942년 대구에서 태어난 고인은 대구의 호텔 나이트클럽에서 기타리스트로 음악 활동을 시작해 1971년 그룹사운드 '윤항기와 키브라더스'에서 활동했다. 1974년에는 그룹사운드 '김훈과 트리퍼스'의 '나를 두고 아리랑'을 작사·작곡했으며, 이듬해 이 노래를 현혜미와 직접 듀엣으로 불러 발표했다. '나를 두고 아리랑'은 '나를 나를 나를 두고/ 물건너 가시더니/ 한 달 두 달 해가 또 가도/ 편지 한 장 없네'라는 서글픈 가사로 인기를 끌어 고인의 대표곡으로 남았다. 박성서 대중음악평론가는 '이 곡은 우리나라 고유의 선율을 당시 유행하던 고고리듬으로 편곡하고 그룹사운드 반주를 붙여 만든 파격적인 노래'라며 '김훈과 트리퍼스는 '나를 두고 아리랑'으로 당시 10대 가수상까지 받았다'고 말했다. 고인과 함께 음악 활동을 한 윤항기는 '1970년대 당시 많은 가수가 팝적인 음악을 추구할 때, 김중신은 한국 전통 선율을 활용한 '나를 두고 아리랑'으로 우리 가요에 변화를 줬다'고 평했다. '나를 두고 아리랑'은 이후 윤항기를 비롯해 나훈아, 조미미, 이용복, 선우성 등 많은 가수가 리메이크한 1970년대 가요계 명곡 가운데 하나로 남았다. '나를 두고 아리랑'은 이후 윤항기를 비롯해 나훈아, 조미미, 이용복, 선우성 등 많은 가수가 리메이크한 1970년대 가요계 명곡 가운데 하나로 남았다. '나를 두고 아리랑'은 이후 윤항기를 비롯해 나훈아, 조미미, 이용복, 선우성 등 많은 가수가 리메이크한 1970년대 가요계 명곡 가운데 하나로 남았다. '나를 두고 아리랑'은 이후 윤항기를 비롯해 나훈아, 조미미, 이용복, 선우성 등 많은 가수가 리메이크한 1970년대 가요계 명곡 가운데 하나로 남았다. '나를 두고 아리랑'은 이후 윤항기를 비롯해 나훈아, 조미미, 이용복, 선우성 등 많은 가수가 리메이크한 1970년대 가요계 명곡 가운데 하나로 남았다. '나를 두고 아리랑'은 이후 윤항기를 비롯해 나훈아, 조미미, 이용복, 선우성 등 많은 가수가 리메이크한 1970년대 가요계 명곡 가운데 하나로 남았다."

def get_gpu_usage():
    GPU_SIZE = 12 * 1024 # MB
    for gpu in GPUtil.getGPUs():
        return gpu.memoryUtil*GPU_SIZE

# @torch.no_grad()
def mem_usage_of_expanded_bert(model, tokenizer, max_token):
    expand_bert_token_size(model, tokenizer, max_token)
    torch.cuda.empty_cache()
    mem_usage = get_gpu_usage()
    model.eval().cuda()
    input_ids = tokenizer(text_content, return_tensors='pt', padding='max_length', truncation=True, max_length=max_token).to(torch.device('cuda'))
    model(**input_ids)
    return get_gpu_usage() - mem_usage

# @torch.no_grad()
def mem_usage_of_long_bert(model, tokenizer, max_token, attn_win):
    convert_bert_with_longformer(model, tokenizer, max_token, attn_win)
    torch.cuda.empty_cache()
    mem_usage = get_gpu_usage()
    model.eval().cuda()
    input_ids = tokenizer(text_content, return_tensors='pt', padding='max_length', truncation=True, max_length=max_token).to(torch.device('cuda'))
    mem_usage = get_gpu_usage()
    model(**input_ids)
    return get_gpu_usage() - mem_usage
    


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=2)
    
    # inputs = tokenizer(text_content, return_tensors='pt')
    # print(inputs.input_ids.shape)
    
    # mem_usage = mem_usage_of_expanded_bert(model, tokenizer, 2048)
    mem_usage = mem_usage_of_long_bert(deepcopy(model), tokenizer, 2048, 32)
    # mem_usage = mem_usage_of_long_bert(deepcopy(model), tokenizer, 2048, 64)
    # mem_usage = mem_usage_of_long_bert(deepcopy(model), tokenizer, 2048, 128)
    # mem_usage = mem_usage_of_long_bert(deepcopy(model), tokenizer, 2048, 256)
    
    print(f'{mem_usage}MB used for inference.')