# -*- coding: utf-8 -*-
from detector.detector import Detector
from summarizer.summarizer import Summarizer

class TrueTitle():
    def __init__(self, detector:str='yeti-s/clickbait_detector', summarizer:str='yeti-s/kobart-title-generator') -> None:
        self.detector = Detector(detector)
        self.summarizer = Summarizer(summarizer)
    
    def create(self, title:str, content:str):
        new_title = ''
        is_reliable = self.detector.detect(title, content)
        if not is_reliable:
            new_title = self.summarizer.summarize(content)
        
        return is_reliable, new_title
    
def unit_test():
    true_title = "'나를 두고 아리랑'의 작곡가 겸 가수 김중신 정정한 80세 잔치"
    true_content = "1970년대 히트곡 '나를 두고 아리랑'을 만들고 부른 작곡가 겸 가수 김중신이 지난 2일(이하 현지시간) 미국 하와이에서 지병으로 별세했다고 박성서 대중음악평론가가 8일 전했다. 향년 81세. 1942년 대구에서 태어난 고인은 대구의 호텔 나이트클럽에서 기타리스트로 음악 활동을 시작해 1971년 그룹사운드 '윤항기와 키브라더스'에서 활동했다. 1974년에는 그룹사운드 '김훈과 트리퍼스'의 '나를 두고 아리랑'을 작사·작곡했으며, 이듬해 이 노래를 현혜미와 직접 듀엣으로 불러 발표했다. '나를 두고 아리랑'은 '나를 나를 나를 두고/ 물건너 가시더니/ 한 달 두 달 해가 또 가도/ 편지 한 장 없네'라는 서글픈 가사로 인기를 끌어 고인의 대표곡으로 남았다. 박성서 대중음악평론가는 '이 곡은 우리나라 고유의 선율을 당시 유행하던 고고리듬으로 편곡하고 그룹사운드 반주를 붙여 만든 파격적인 노래'라며 '김훈과 트리퍼스는 '나를 두고 아리랑'으로 당시 10대 가수상까지 받았다'고 말했다. 고인과 함께 음악 활동을 한 윤항기는 '1970년대 당시 많은 가수가 팝적인 음악을 추구할 때, 김중신은 한국 전통 선율을 활용한 '나를 두고 아리랑'으로 우리 가요에 변화를 줬다'고 평했다. '나를 두고 아리랑'은 이후 윤항기를 비롯해 나훈아, 조미미, 이용복, 선우성 등 많은 가수가 리메이크한 1970년대 가요계 명곡 가운데 하나로 남았다."
    true_title = TrueTitle()
    print(true_title.create(true_title, true_content))
    
if __name__ == '__main__':
    unit_test()