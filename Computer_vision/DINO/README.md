# DINO를 활용한 유사한 이미지 검색

DINO의 경우 distillation 기법과 self-supervised learning을 기반으로 이미지의 인식하는 능력을 키우는 Foundation model의 일종입니다.

DINO의 경우 이미지를 특징에 따라서 특정 feature vector로 변환해 주기에 이러한 feature vector를 기반으로 비슷한 이미지를 검색하도록 코드를 구현하였습니다.

DNIO를 기반으로 이미지를 임베딩 하고, 각각의 이미지 임베딩 값을 파일명을 기준으로 FAISS에 저장해줍니다. 이떄 FAISS는 코싸인 유사도와 L2 를 모두활용하도록 설정하였습니다.

그리고 FAISS에서 제공하는 search()를 활용해서 입력 이미지와 유사한 vector를 추출하였습니다.

실험 결과 cosein과 L2는 크게 영향을 주지 못하는것으로 확인되었습니다.

+) 추가적으로 feature vector의 크기가 영향을 주는지 알아보고 싶다고 생각됩니다.