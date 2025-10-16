# make_test_topic.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# 경로만 바꿔 써
TRAIN_CSV = "/workspace/NLP_Dialogue_Summarization/data/train.csv"
TEST_CSV  = "/workspace/NLP_Dialogue_Summarization/data/test.csv"
OUT_CSV   = "/workspace/NLP_Dialogue_Summarization/data/test_topic.csv"

def main():
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    # 최소 컬럼 체크
    assert "dialogue" in train.columns and "topic" in train.columns, "train.csv에 'dialogue','topic' 필요"
    assert "dialogue" in test.columns, "test.csv에 'dialogue' 필요"

    # 간단/빠른 파이프라인 (한글에 강한 문자 n-gram)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_features=200000)),
        ("clf", LinearSVC())
    ])

    # 학습
    pipe.fit(train["dialogue"], train["topic"])

    # 예측
    test_topic = pipe.predict(test["dialogue"])

    # 저장 (원본 test에 topic 추가)
    out = test.copy()
    out["topic"] = test_topic
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved: {OUT_CSV}")
    print(out.head())
    


def build_topic_classifier(train_df):
    """TF-IDF 기반 topic classifier 생성"""
    topic_texts = {}
    
    # 각 topic별로 모든 대화를 합침
    for topic in train_df['topic'].unique():
        topic_dialogues = train_df[train_df['topic'] == topic]['dialogue'].tolist()
        topic_texts[topic] = ' '.join(topic_dialogues)
    
    # TF-IDF 벡터라이저 학습
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),  # 1-gram과 2-gram 사용
        min_df=2  # 최소 2번 이상 등장한 단어만 사용
    )
    
    topics = list(topic_texts.keys())
    topic_vectors = vectorizer.fit_transform([topic_texts[t] for t in topics])
    
    return vectorizer, topic_vectors, topics

def predict_topic(dialogue, vectorizer, topic_vectors, topics):
    """코사인 유사도로 가장 유사한 topic 반환"""
    dialogue_vector = vectorizer.transform([dialogue])
    similarities = cosine_similarity(dialogue_vector, topic_vectors)[0]
    
    best_topic_idx = np.argmax(similarities)
    best_score = similarities[best_topic_idx]
    
    return topics[best_topic_idx], best_score

def add_topics_to_test(train_csv_path='train.csv', test_csv_path='test.csv', output_csv_path='test_topic.csv'):
    """
    Test 데이터에 topic 컬럼을 추가하여 새로운 CSV 파일로 저장
    
    Args:
        train_csv_path: 학습 데이터 경로 (topic 컬럼 포함)
        test_csv_path: 테스트 데이터 경로 (topic 컬럼 없음)
        output_csv_path: 출력 파일 경로 (topic 컬럼 추가됨)
    """
    print(f"Loading train data from {train_csv_path}...")
    train_df = pd.read_csv(train_csv_path)
    
    print(f"Loading test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    
    print("Building topic classifier...")
    vectorizer, topic_vectors, topics = build_topic_classifier(train_df)
    
    print(f"Available topics: {topics}")
    print(f"Predicting topics for {len(test_df)} test samples...")
    
    predictions = []
    confidence_scores = []
    
    for idx, dialogue in enumerate(test_df['dialogue']):
        topic, score = predict_topic(dialogue, vectorizer, topic_vectors, topics)
        predictions.append(topic)
        confidence_scores.append(score)
        
        # 진행상황 출력 (100개마다)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples...")
    
    # Topic 컬럼 추가
    test_df['topic'] = predictions
    test_df['topic_confidence'] = confidence_scores
    
    # 결과 저장
    print(f"\nSaving results to {output_csv_path}...")
    test_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    # 통계 정보 출력
    print("\n" + "="*60)
    print("Topic Distribution:")
    print(test_df['topic'].value_counts())
    print("\n" + "="*60)
    print("Confidence Statistics:")
    print(f"Mean confidence: {test_df['topic_confidence'].mean():.4f}")
    print(f"Min confidence: {test_df['topic_confidence'].min():.4f}")
    print(f"Max confidence: {test_df['topic_confidence'].max():.4f}")
    
    # 낮은 confidence 샘플 확인
    low_confidence_threshold = 0.1
    low_confidence = test_df[test_df['topic_confidence'] < low_confidence_threshold]
    print(f"\nLow confidence predictions (< {low_confidence_threshold}): {len(low_confidence)}")
    
    if len(low_confidence) > 0:
        print("\nSample low confidence predictions:")
        print(low_confidence[['dialogue', 'topic', 'topic_confidence']].head(3))
    
    print("\n" + "="*60)
    print(f"✅ Successfully saved to {output_csv_path}")
    print("="*60)
    
    return test_df

def main2():
    test_with_topics = add_topics_to_test(
        train_csv_path=TRAIN_CSV,
        test_csv_path=TEST_CSV, 
        output_csv_path=OUT_CSV
    )

if __name__ == "__main__":
    main2()

