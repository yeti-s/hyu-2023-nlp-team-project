# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer():
    def __init__(self, model_name:str='yeti-s/kobart-news-summarization', max_length:int=128, device=torch.device('cuda')) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.device = device
        self.model.eval().to(device)
    
    @torch.no_grad()
    def summarize(self, content:str) -> str:
        input_ids = self.tokenizer(content, return_tensors='pt').to(self.device)
        gen_ids = self.model.generate(**input_ids, max_length=self.max_length, use_cache=True)
        generated = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return generated
    
    
def unit_test():
    content = '첫 방송통신위원장인 이동관 위원장은 취임 3주 만에 가짜뉴스 근절 방안을 발표했습니다. 이어 방송통신심의위원회도 "가짜뉴스 심의전담센터"를 출범시켰고, 인터넷 언론까지 심의를 확대하겠다는 방침을 밝혔습니다. 가짜뉴스 대책은 한 달여 사이 언론계에 엄청난 파장을 불러왔습니다. 법적 근거도 빈약하고, 개념도 분명치 않은 가짜뉴스 대응이 언론자유를 크게 위축시킬 것이라는 비판이 나왔지만, 방통위와 방심위는 가짜뉴스 대응의 필요성을 강조하며 속도를 늦추지 않고 있습니다. 최근 방통위는 가짜뉴스 근절 추진 현황과 해외 사례라는 자료를 만들어 국무회의에 보고했습니다. KBS는 가짜뉴스의 개념부터 쟁점, 근절 계획까지 담긴 보고서 내용을 꼼꼼히 살펴봤습니다. 분석 결과는 3번에 걸쳐 정리합니다.'
    summarizer = Summarizer(max_length=128)
    print(summarizer.summarize(content))
    
if __name__ == '__main__':
    unit_test()