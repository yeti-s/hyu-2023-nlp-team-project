# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer
from LongBERT import LongBertForSequenceClassification
class Detector():
    def __init__(self, model_name:str='yeti-s/longbert-clickbait-detector', device=torch.device('cuda')) -> None:
        self.model = LongBertForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.eval().to(device)
    
    @torch.no_grad()
    def detect(self, title:str, content:str) -> str:
        input_ids = self.tokenizer(title, content, return_tensors='pt', padding='max_length', truncation=True, max_length=1024).to(self.device)
        outputs = self.model(**input_ids)
        logits = outputs.logits
        pred = torch.argmax(torch.nn.functional.softmax(logits), dim=1)
        return pred.item()
 
##############################################################
########################## TEST ##############################
##############################################################

def unit_test():
    true_title = 'LK-99 초전도체 아님'
    true_content = '연구진이 상온상압 초전도체라고 주장했던 LK-99에 대해 한국초전도저온학회 검증위원회가 "원논문의 데이터와 국내외 재현실험연구결과를 종합해 고려해 보면, LK-99 가 상온 상압 초전도체라는 근거는 전혀 없다"고 13일 밝혔다. 검증위는 이날 그동안의 국내외 검증 시도를 종합해 백서를 발간, 온라인을 통해 배포했다. 백서에 따르면 그동안 국내 8개 연구소에서 LK-99 관련 논문 저자들이 제시한 방법에 따라 재현연구를 한 결과, 상온 또는 저온에서 초전도성을 보인 결과는 없었다. 검증위는 특히 "국내외 재현 실험 연구에서 저항 0과 마이스너 효과를 보여주는 경우는 없었다"며 "대부분의 결과는 LK-99 가 오히려 비저항 값이 매우 큰 부도체임을 보여주고 있다"고 설명했다. 검증위는 또 퀀텀에너지연구소의 LK-99 시료를 제공받아 교차 측정하고자 했으나, 연구소로부터 시료는 제공되지 않았고 지금까지 교차측정을 통한 검증은 이뤄지지 못했다고 덧붙였다.'
    detector = Detector()
    print(detector.detect(true_title, true_content))
    
if __name__ == '__main__':
    unit_test()