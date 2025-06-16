# 통합된 하이브리드 향수 추천 시스템 (FAISS + Neo4j + 평가 + Streamlit)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score
import faiss
import random
from collections import defaultdict
from neo4j import GraphDatabase
import streamlit as st

st.set_page_config(page_title="향수 추천 시스템", layout="wide")

# ------------------------
# 캐싱된 모델 및 인덱스 로딩
# ------------------------
@st.cache_data
def load_dataframe():
    csv_url = "https://raw.githubusercontent.com/rawanalqarni/Perfumes_Recommender/main/Datasets/Perfume_Dataset.csv"
    df = pd.read_csv(csv_url)
    return df[['Name', 'Description','Top_note', 'Middle_note', 'Base_note']].fillna('')

df = load_dataframe()

@st.cache_resource
def load_tfidf_vectors(df):
    vectorizer = TfidfVectorizer()
    top_vec = vectorizer.fit_transform(df['Top_note']).toarray()
    middle_vec = vectorizer.fit_transform(df['Middle_note']).toarray()
    base_vec = vectorizer.fit_transform(df['Base_note']).toarray()
    max_dim = max(top_vec.shape[1], middle_vec.shape[1], base_vec.shape[1])
    pad = lambda arr, dim: np.pad(arr, ((0, 0), (0, dim - arr.shape[1])), 'constant')
    top_vec = pad(top_vec, max_dim)
    middle_vec = pad(middle_vec, max_dim)
    base_vec = pad(base_vec, max_dim)
    note_vectors = (0.3 * top_vec + 0.5 * middle_vec + 0.2 * base_vec).astype('float32')
    return note_vectors

note_vectors = load_tfidf_vectors(df)

@st.cache_resource
def load_sbert_embeddings(df):
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    return model.encode(df['Description'].tolist(), normalize_embeddings=True)

desc_vecs = load_sbert_embeddings(df)

@st.cache_resource
def build_faiss_index():
    combined = np.hstack([note_vectors]).astype('float32')
    faiss.normalize_L2(combined)
    index = faiss.IndexFlatIP(combined.shape[1])
    index.add(combined)
    return index, combined

index, combined = build_faiss_index()

# FAISS 기반 추천
@st.cache_data
def recommend_perfumes_faiss(user_id, top_k=30):
    liked = get_user_liked_perfumes(user_id)
    liked_idxs = [df[df['Name'] == p].index[0] for p in liked if p in df['Name'].values]
    if not liked_idxs:
        return []
    query = np.mean([combined[i] for i in liked_idxs], axis=0).reshape(1, -1)
    faiss.normalize_L2(query)
    _, idxs = index.search(query, top_k)
    recs = [df.iloc[i]['Name'] for i in idxs[0]]
    return [r for r in recs]

# 사용자 생성 및 유사 사용자 기반 추천 (Neo4j)
user_ids = [f"user_{i}" for i in range(1, 501)]
uri = "neo4j+s://c5aa6fa9.databases.neo4j.io"
username = "neo4j"
password = "lqdv0Rvs20n5MOUdonw39z15fyM6CPagVtc18vxzbmw"

@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(uri, auth=(username, password))

driver = get_neo4j_driver()

def get_user_liked_perfumes(user_id):
    query = """
    MATCH (u:User {userId: $user_id})-[:LIKES]->(p:Perfume)
    RETURN p.name AS perfume_name
    """
    with driver.session() as session:
        result = session.run(query, user_id=user_id)
        return [record["perfume_name"] for record in result]

def get_recommendations_for_user(user_id):
    query = """
    MATCH (me:User {userId: $user_id})-[:LIKES]->(p1)<-[:LIKES]-(other:User)-[:LIKES]->(p2)
    RETURN p2.name AS perfume_name, COUNT(DISTINCT other) AS score
    ORDER BY score DESC
    """
    with driver.session() as session:
        result = session.run(query, user_id=user_id)
        return pd.DataFrame([r.data() for r in result])

# 하이브리드 추천
def get_hybrid_recommendations(user_id, top_k=10, alpha=0.5, exclude_liked=False):
    content_recs = recommend_perfumes_faiss(user_id, top_k=30)
    collab_df = get_recommendations_for_user(user_id)
    content_score = {p: 30 - i for i, p in enumerate(content_recs)}
    collab_score = {row['perfume_name']: row['score'] for _, row in collab_df.iterrows()}
    all_names = set(content_score) | set(collab_score)
    liked = get_user_liked_perfumes(user_id)
    hybrid_scores = {n: alpha * content_score.get(n, 0) + (1 - alpha) * collab_score.get(n, 0) for n in all_names}
    sorted_perfumes = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    recs = [name for name, _ in sorted_perfumes]
    return recs[:top_k]

# 평가 지표 계산
def evaluate_user(gt, pred, all_items, popularity_dict):
    y_true = [1 if i in gt else 0 for i in all_items]
    y_pred = [1 if i in pred else 0 for i in all_items]
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    novelty = np.mean([-np.log2((popularity_dict.get(p, 0) + 1) / len(user_ids)) for p in pred])
    return precision, recall, novelty

def get_user_preferred_family(user_id):
    query = """
    MATCH (u:User {userId: $user_id})
    RETURN u.preferred_family AS family
    """
    with driver.session() as session:
        result = session.run(query, user_id=user_id)
        record = result.single()
        return record["family"] if record else "정보 없음"


# Streamlit 인터페이스
st.title("🌸 하이브리드 향수 추천 시스템")

selected_user = st.selectbox("사용자를 선택하세요", user_ids)
top_k = st.slider("Top-K 추천 개수", 5, 30, 10)
alpha = st.slider("콘텐츠 기반 가중치 (0=협업, 1=콘텐츠)", 0.0, 1.0, 0.5)

# 사용자가 선호하는 향수 계열 정보 출력
preferred_family = get_user_preferred_family(selected_user)
st.markdown(f"#### 👤 **User ID:** {selected_user}")
st.markdown(f"#### 🌿 **Preferred Fragrance Family:** `{preferred_family}`")

st.subheader("💖 사용자가 좋아했던 향수")

liked_perfumes = get_user_liked_perfumes(selected_user)
liked_df = df[df['Name'].isin(liked_perfumes)][['Name', 'Top_note', 'Middle_note', 'Base_note', 'Description']]

# 표 형식으로 출력
st.dataframe(liked_df.reset_index(drop=True), use_container_width=True)

if st.button("추천 받기"):
    with st.spinner("추천 생성 중..."):
        recs = get_hybrid_recommendations(selected_user, top_k=top_k, alpha=alpha)
        st.subheader("✨ 추천된 향수 목록")

        recommended_df = df[df['Name'].isin(recs)][['Name', 'Top_note', 'Middle_note', 'Base_note', 'Description']]

        # 표 형식으로 출력
        st.dataframe(recommended_df.reset_index(drop=True), use_container_width=True)

        gt = get_user_liked_perfumes(selected_user)
        all_perfumes = df['Name'].tolist()
        perfume_popularity = {name: 1 for name in all_perfumes}
        precision, recall, novelty = evaluate_user(gt, recs, all_perfumes, perfume_popularity)
        st.markdown("### 📊 평가 지표")
        st.markdown(f"**Precision:** {precision:.3f}")
        st.markdown(f"**Recall:** {recall:.3f}")
        st.markdown(f"**Novelty:** {novelty:.3f}")
