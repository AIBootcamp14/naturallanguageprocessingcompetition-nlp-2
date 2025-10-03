import pandas as pd
import os
import time
from tqdm import tqdm
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.
from openai import OpenAI # openai==1.2.0

DATA_PATH = "/workspace/NLP_Dialogue_Summarization/data/"
# RESULT_PATH = "/workspace/NLP_Dialogue_Summarization/output/submission/"
    
    
def set_api():
    UPSTAGE_API_KEY = "up_YlGro1bbSlo1E34E78sQFKoQOUy0w" # upstage.ai에서 발급받은 API KEY를 입력해주세요.

    client = OpenAI(
        api_key=UPSTAGE_API_KEY,
        base_url="https://api.upstage.ai/v1/"
    )

    return client

def build_prompt(dialogue): 
    
    # Few-shot prompt를 생성하기 위해, train data의 일부를 사용합니다.
    # 항상 같은 5개 샘플을 뽑기 위해 random_state 고정
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    few_shot_samples = train_df.sample(
        n=min(5, len(train_df)), 
        random_state=42
    ).reset_index(drop=True)

    # 샘플 5개를 프롬프트에 나열
    samples_block = []
    for i, row in few_shot_samples.iterrows():
        samples_block.append(
            f"Sample Dialogue {i+1}:\n{row['dialogue']}\n\n"
            f"Sample Topic {i+1}:\n{row['topic']}\n"
        )
    samples_block = "\n".join(samples_block)

    system_prompt = """# Role
            You are a search optimization specialist with expertise in topic keyword extraction and information retrieval.

            # Instructions
            Extract 1 main topic keyword from the dialogue and return it without any explanation:
            1. [primary keyword]

            Focus on the most important search terms for optimal information retrieval."""

    user_prompt = (
        "Following the instructions below, summarize the given document.\n"
        "Instructions:\n"
        "1. Read the dialogue carefully.\n"
        "2. Extract exactly 1 main topic keyword that best represents the dialogue.\n"
        "3. Output only the keyword, without any explanation or extra text.\n"
        "4. If the topic is unclear, output: unknown\n\n"
        # 고정 샘플 제공
        f"{samples_block}\n"
        "Dialogue:\n"
        f"{dialogue}\n\n"
        "Topic:\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    
def extract_topic(client, dialogue):
    summary = client.chat.completions.create(
        model="solar-pro2",
        messages=build_prompt(dialogue),
        temperature=0.3,
        top_p=0.5,
    )

    return summary.choices[0].message.content
    
    
def test_on_train_data(client, df, num_samples=3):
    for idx, row in df[:num_samples].iterrows():
        dialogue = row['dialogue']
        extracted_topic = extract_topic(client, dialogue)
        print(f"Dialogue:\n{dialogue}\n")
        print(f"Pred Topic: {extracted_topic}\n")
        print(f"Gold Topic: {row['topic']}\n")
        print("=="*50)
        
def add_topic_to_test_data(client, df):
    topics = []
    confidence_scores = []
    
    start_time = time.time()
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        dialogue = row['dialogue']
        extracted_topic = extract_topic(client, dialogue)
        topics.append(extracted_topic)
        confidence_scores.append(1.0)

        if (idx + 1) % 100 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time + 5
                print(f"Elapsed time: {elapsed_time:.2f} sec")
                print(f"Waiting for {wait_time} sec")
                time.sleep(wait_time)
            
            start_time = time.time()

    df['topic'] = topics
    df['topic_confidence'] = confidence_scores
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    df.to_csv(os.path.join(DATA_PATH, "test_topic_solar.csv"), index=False)
    
    return df    




def main():
    train_df = pd.read_csv(os.path.join(DATA_PATH,'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_PATH,'test.csv'))
    
    client = set_api()
    # test_on_train_data(client, train_df)
    add_topic_to_test_data(client, test_df)
      

if __name__ == "__main__":
    main()