# auto profiling model
# 평균 성능 accuray 약 98.5%
    
    KISA 검증       

    1. 실행 방법 (dti_v3_total_model.py)
        1.1	Terminal에서 해당 py 파일이 들어있는 디렉토리로 이동
            1)	bash
            2)	[ex] cd /home/ctilab/dti_v3/test2/git_clone/dti

        2.1 파이썬 환경에서 해당 py 파일 실행
            1) python3.7 dti_v3_total_model.py -m[모델번호] -[학습/예측 여부] -[바로 실행 여부]
            2) 모델번호 : 335
            3) 학습/예측 여부 : 학습(t), 예측(p)
            4) 바로 실행시 n 입력
            
    2. 학습 실행
        2.1 버전 세팅
            1) '연_월_일' 형태로 지정
            2) [ex] 2021_01_10        
            
        2.2 config  로드
            1) 현재 연동되어 있는 DTI 내부의 AI모델 관리 > AI모델 설정 > 해당되는 모델번호 선택
            2) 관련내용 수정 가능 및 해당 내용으로 모델에 반영됨
            
        2.3 데이터 불러오기
            1) config에 설정 된 normal_days와 attack_days 날짜를 기준으로 공격 리스트별 데이터 불러오기 진행
            2) limit 개수가 정해져있으며, 각 공격종류 및 정상 데이터마다 최대 limit 개수 까지만큼의 데이터 불러오기 진행
            2) 공격 리스트 = ['normal', 'SQL_INJECTION', 'XSS', 'BEACONING', 'CREDENTIAL']
            
        2.4 데이터 전처리
            1) 정수형, 카테고리형, 문자형으로 데이터를 구분하여 전처리 진행
            2) 정수형 데이터
                : Minmax scaling를 활용한 scaler fitting 후 pickle 형식으로 자동 저장
            3) 카테고리형 데이터 
                : one-hot encoding fitting 후 pickle 형식으로 자동 저장
            4) 문자형 데이터
                : tf-idf fitting 후 pickle 형식으로 자동 저장                        
            
        2.5 모델학습 (Fitting & Save Model)
            1) Decision Tree 우선적으로 진행
                : model_config에서 max_depth 설정 가능
            2) Decision Tree Fitting 후 pickle 형식으로 자동 저장
            3) CNN 모델 fitting
                : 정상을 제외한 공격 종류 별 개별 학습 진행 (총 4개)
            4) 모델 Fitting 후 h5 형식으로 자동 저장 (저장 경로 - auto_profiling_model/버전)
            
    3. 예측 실행
        3.1 버전 세팅
            1) 학습 된 버전 중 가장 최신 버전으로 자동 설정
            2) '연_월_일' 형태로 지정
            
        3.2 config  로드
            1) 현재 연동되어 있는 DTI 내부의 AI모델 관리 > AI모델 설정 > 해당되는 모델번호 선택
            2) 관련내용 수정 가능 및 해당 내용으로 모델에 반영됨
            
        3.3 데이터 전처리
            1) 지정 된 버전에서 학습 된 전처리 모듈 호출 (tf-idf, Minmaxx scaler 등)
                : 문자열 처리(tf_idf), Minmax Scaling, one-hot encoder
            2) 각 모듈 transform 진행
            
        3.4 모델예측
            1) 데이터 예측 후 검증 시행
            2) 검증은 Accuracy score와 Confusion Matrix 점수로 확인
            3) 검증 완료 후 데이터 Insert (DB)
            