# Perfume Recommender System
향수 취향에 맞는 제품을 추천해주는 하이브리드 추천 시스템입니다.
**향수 노트(TOP, MIDDLE, BASE)** 와 **설명(description)** 을 기반으로 한 콘텐츠 기반 필터링(Content-Based Filtering)과
사용자 간 유사도를 활용한 협업 필터링(Collaborative Filtering)을 결합하여 정교한 추천을 제공합니다.

## 프로젝트 요약

- 향수 데이터 전처리 및 TF-IDF / BERT 임베딩 적용
- FAISS를 활용한 향수 간 유사도 계산
- Neo4j 기반 그래프에서 사용자-향수 관계 분석 및 협업 필터링
- 추천 성능 평가 (Precision / Recall / Novelty)

<br/>

🌐 [향수 추천 사이트](https://sforseohn-perfume-recommender-system-streamlit-app-t4x81r.streamlit.app/)

👉 [향수 추천 시스템 발표자료 보기](https://drive.google.com/file/d/1NjS12g4lnucf9WV8anMQXoDLbNIyvy88/view?usp=sharing)


## 주요 기술 스택
Python, Pandas, NumPy

Scikit-learn, FAISS, SentenceTransformer

Neo4j, py2neo

Streamlit (웹 인터페이스)

<br/>
