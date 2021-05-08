### training
* python train_trainer.py --config_name [config(json)]
* ex) python train_trainer.py --config_name config_example

### inference
* python inference.py --config_name [config(json)] --ckpt_name [saved_model_ckpt]
* ex) python inference.py --config_name config_example --ckpt_name checkpoint-1000

### ensemble
* 코드 상에서 경로 변경 후 앙상블 (soft vote, hard vote)

### QA task
* dataset&evaulate_QA : QA task를 위한 데이터 셋 생성과 pipeline을 활용한 간단한 테스트
* train_QA : QA dataset을 활용하여 학습
* inference_QA : 실제 inferece하지 않고 한 테스트 데이터에 대해 질문 별 결과만 출력한다. 
  (결과 확인 후 성능이 좋지 않아 사용 x)
