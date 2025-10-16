# src/data/augment_data.py
import os
import time
import json
import re
from typing import Optional, List, Dict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tqdm import tqdm


def extract_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def clean_dialogue_output(text: str) -> str:
    """Normalize dialogue formatting: remove blank lines, enforce '#PersonX#: ' lines."""
    text = re.sub(r'(?<!^)\s*(#Person\d+#:)', r'\n\1', text)
    normalized = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = re.match(r"^#*Person(\d+)#*[:：]?\s*(.*)", line)
        if m:
            speaker = m.group(1)
            utterance = m.group(2).strip()
            line = f"#Person{speaker}#: {utterance}".strip()
        else:
            line = re.sub(r'^\((#Person\d+#)\)\s*', r'\1: ', line)

        line = re.sub(r'\s+', ' ', line)
        normalized.append(line)

    return '\n'.join(normalized)


def _dialogue_to_entries(text: str) -> List[Dict[str, str]]:
    entries = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r'^(#Person\d+#)\s*[:：]\s*(.*)$', line)
        if not m:
            continue
        entries.append(
            {"speaker": m.group(1), "utterance": m.group(2).strip()})
    return entries


def entries_to_dialogue(entries: List[Dict[str, str]]) -> str:
    lines = []
    for item in entries:
        speaker = item.get("speaker", "").strip()
        utterance = item.get("utterance", "").strip()
        if not speaker or not utterance:
            continue
        lines.append(f"{speaker}: {utterance}")
    return '\n'.join(lines)


def augment_paraphrase(
    df: pd.DataFrame,
    num_samples: int,
) -> pd.DataFrame:
    """의미 유지형 재작성(paraphrase)으로 데이터를 증강합니다."""
    print(
        f"\n--- Starting Paraphrase Augmentation for {num_samples} samples ---")
    if num_samples == 0:
        return pd.DataFrame()

    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        print("  UPSTAGE_API_KEY not found. Skipping paraphrase augmentation.")
        return pd.DataFrame()

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.upstage.ai/v1",
    )

    samples = df.sample(n=min(num_samples, len(df)), random_state=42)

    system_prompt = """
    당신은 한국어 대화와 요약문을 자연스럽게 재작성하는 전문가입니다.
    원문의 사실과 의도, 화자 태그(#Person1#, #Person2# 등)를 반드시 유지하되, 표현은 자연스럽고 새로운 문장으로 바꾸세요.
    출력은 JSON 형식으로만 제공합니다.
    """

    augmented_rows: list[dict] = []

    for _, row in tqdm(samples.iterrows(), total=len(samples), desc="Paraphrasing"):
        dialogue = row.get('dialogue', '')
        summary = row.get('summary', '')
        entries = _dialogue_to_entries(dialogue)
        if not entries:
            continue
        original_tags = {entry['speaker'] for entry in entries}
        dialogue_json = json.dumps(entries, ensure_ascii=False, indent=2)

        user_prompt = f"""아래 규칙을 참고하여 다음 대화와 요약을 의미는 유지하면서 더 자연스럽고 새로운 표현으로 다시 작성하세요.

        ### 원본 대화 (JSON)
        {dialogue_json}

        ### 원본 요약
        {summary}

        [지침]
        1. `speaker` 값은 반드시 원본에 존재하는 태그(`#PersonX#`)만 사용하고, 다른 태그를 추가하지 마세요.
        2. 각 발화는 `dialogue` 배열의 요소로, 한 줄 한 화자 방식(`speaker`, `utterance`)을 유지하세요.
        3. 중요한 사실, 수치, 고유명사를 빠뜨리지 말고, 필요 시 자연스럽게 재배열만 허용됩니다.
        4. 요약은 대화 내용을 정확히 반영하되, 문장 구조와 어휘를 새롭게 바꾸세요.
        5. 전체 길이는 원문과 비슷하게 유지하되, 동일 문장을 반복하지 마세요.
        6. 출력은 아래 예시처럼 JSON 구조로만 작성하세요.
        {{"dialogue": [{{"speaker": "#Person1#", "utterance": "..."}}], "summary": "..."}}
        """

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="solar-1-mini-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.6,
                    max_tokens=1800,
                )
                content = response.choices[0].message.content.strip()
                data = extract_json(content)
                if not data:
                    raise ValueError("Invalid JSON response")

                new_dialogue_entries = data.get('dialogue', [])
                if not isinstance(new_dialogue_entries, list):
                    raise ValueError("Invalid dialogue structure")
                new_tags = {entry.get('speaker', '')
                            for entry in new_dialogue_entries}
                if new_tags != original_tags:
                    raise ValueError(
                        "Speaker tags mismatch between original and paraphrase")

                new_dialogue = entries_to_dialogue(new_dialogue_entries)
                new_summary = data.get('summary', '').strip()

                if not new_dialogue or not new_summary:
                    raise ValueError("Empty dialogue or summary in response")

                new_row = row.to_dict()
                new_row['fname'] = f"{row['fname']}-rp"
                new_row['dialogue'] = new_dialogue
                new_row['summary'] = new_summary
                augmented_rows.append(new_row)
                break
            except RateLimitError:
                wait_time = 2 ** attempt
                print(f"  Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                if attempt == 2:
                    print(
                        f"  Solar paraphrase failed for {row.get('fname')}: {e}")
                else:
                    print(f"  Error during paraphrase: {e}. Retrying...")
                time.sleep(1)
        else:
            continue

    print("--- Paraphrase augmentation complete ---")
    return pd.DataFrame(augmented_rows)


def augment_speaker_swap(df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """#Person1#과 #Person2#의 역할을 바꾸어 데이터를 증강합니다."""
    print(f"\n--- Starting Speaker Swap for {num_samples} samples ---")
    if num_samples == 0:
        return pd.DataFrame()

    samples = df[
        df['dialogue'].str.contains('#Person1#')
        & df['dialogue'].str.contains('#Person2#')
        & ~df['dialogue'].str.contains('#Person3#')
    ]
    samples = samples.sample(n=min(num_samples, len(samples)), random_state=42)

    augmented_data = []

    for _, row in tqdm(samples.iterrows(), total=len(samples), desc="Swapping speakers"):
        dialogue = row['dialogue']
        summary = row['summary']

        swapped_dialogue = dialogue.replace('#Person1#', '#TEMP#').replace(
            '#Person2#', '#Person1#').replace('#TEMP#', '#Person2#')
        swapped_summary = summary.replace('#Person1#', '#TEMP#').replace(
            '#Person2#', '#Person1#').replace('#TEMP#', '#Person2#')

        new_row = row.to_dict()
        new_row['fname'] = f"{row['fname']}_ss"
        new_row['dialogue'] = swapped_dialogue
        new_row['summary'] = swapped_summary
        augmented_data.append(new_row)

    print("--- Speaker Swap complete ---")
    return pd.DataFrame(augmented_data)


def generate_synthetic_data(
    df: pd.DataFrame,
    num_samples: int,
    topics_with_weights: pd.Series,
) -> pd.DataFrame:
    """Solar API를 사용하여 새로운 (대화, 요약) 쌍을 생성합니다."""
    print(
        f"\n--- Starting Synthetic Data Generation for {num_samples} samples ---")
    if num_samples == 0:
        return pd.DataFrame()

    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        print(
            "  UPSTAGE_API_KEY not found in .env file. Skipping synthetic data generation.")
        return pd.DataFrame()

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.upstage.ai/v1"
    )

    synthetic_data = []

    for i in tqdm(range(num_samples), desc="Generating synthetic data"):
        few_shot_count = min(2, len(df))
        if few_shot_count > 0:
            few_shot_samples = df.sample(n=few_shot_count)
        else:
            few_shot_samples = pd.DataFrame(columns=df.columns)

        example_items = []
        for _, shot in few_shot_samples.iterrows():
            example_items.append({
                "dialogue": _dialogue_to_entries(shot['dialogue']),
                "summary": shot['summary']
            })
        example_json = json.dumps(example_items, ensure_ascii=False, indent=2)

        new_topic = topics_with_weights.sample(
            n=1, weights=topics_with_weights).index[0]

        prompt = f"""### 지시:
        당신은 한국어 대화 시나리오 생성 전문가입니다. 아래 규칙과 JSON 예시를 참고하여, 주어진 주제에 대한 새로운 대화문과 그에 맞는 요약문을 JSON으로 생성하세요.

        [규칙]
        1. 대화문은 반드시 '#Person1#'과 '#Person2#' 두 명의 화자로 구성되어야 하며, `dialogue` 배열의 각 요소는 `{{"speaker": "#Person1#", "utterance": "..."}}` 형태여야 합니다.
        2. 화자 태그는 '#Person1#', '#Person2#'만 사용하고, '#PersonX#:' 형식을 유지하며 추가 기호를 붙이지 마세요.
        3. 요약문에서도 화자를 언급할 때 반드시 대화문에 등장한 실제 이름만 사용하거나, 이름이 없다면 '#Person1#'과 '#Person2#' 태그를 그대로 사용하세요. 태그와 실제 이름을 섞지 마세요.
        4. 요약문은 `summary` 항목에 문자열로 작성하고, 대화의 핵심 내용을 담도록 합니다.

        ### JSON 예시
        {example_json}

        ---

        ### 주제
        {new_topic}

        ### 출력 형식
        {{"dialogue": [{{"speaker": "#Person1#", "utterance": "..."}}, ...], "summary": "..."}}
        """

        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model="solar-1-mini-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                )
                generated_text = response.choices[0].message.content

                data = extract_json(generated_text)
                if not data:
                    raise ValueError("Invalid JSON response")
                dialogue_list = data.get('dialogue', [])
                if not isinstance(dialogue_list, list) or len(dialogue_list) == 0:
                    raise ValueError("Invalid dialogue structure")
                for entry in dialogue_list:
                    speaker = entry.get('speaker', '')
                    if speaker not in {"#Person1#", "#Person2#"}:
                        raise ValueError(
                            "Unexpected speaker tag in synthetic dialogue")
                summary_part = data.get('summary', '').strip()
                summary_tags = {
                    tag.group() for tag in re.finditer(r'#Person\d+#', summary_part)
                }
                if summary_tags not in ({"#Person1#", "#Person2#"}, set()):
                    raise ValueError(
                        "Summary must use either both speaker tags or none (use real names).")
                new_dialogue = entries_to_dialogue(dialogue_list)

                if new_dialogue and summary_part:
                    new_row = {
                        'fname': f'synthetic_{i}',
                        'dialogue': new_dialogue,
                        'summary': summary_part,
                        'topic': new_topic
                    }
                    synthetic_data.append(new_row)
                    break
            except RateLimitError:
                wait_time = 2 ** attempt
                print(f"  Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"  API call failed with error: {e}. Retrying...")
                time.sleep(1)
        else:
            print(
                f"  Failed to generate data for sample {i} after multiple retries.")

    print("--- Synthetic Data Generation complete ---")
    return pd.DataFrame(synthetic_data)


def run_augmentation(
    original_data_path: str,
    dev_data_path: str,
    augmented_data_path: str,
    num_paraphrase: int,
    num_speaker_swap: int,
    num_synthetic: int,
) -> dict[str, str]:
    """Execute augmentation steps and save each result as a separate CSV.

    Returns a dict mapping augmentation type (keys: 'rp', 'ss', 'synthetic', 'original') to saved CSV path.
    """

    print(f"Loading original data from {original_data_path}")
    original_df = pd.read_csv(original_data_path)
    dev_df = pd.read_csv(dev_data_path)

    all_topics_df = pd.concat([original_df['topic'], dev_df['topic']]).dropna()
    topic_weights = all_topics_df.value_counts(normalize=True)
    print(
        f"Total {len(topic_weights)} unique topics will be used for synthetic data generation with weights.")

    augmented_dir = os.path.dirname(augmented_data_path) or '.'
    os.makedirs(augmented_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(augmented_data_path))[0]

    saved_paths: dict[str, str] = {}

    def _load_existing(suffix: str) -> Optional[pd.DataFrame]:
        output_path = os.path.join(augmented_dir, f"{base_name}_{suffix}.csv")
        if os.path.exists(output_path):
            print(f"{suffix} data already exists. Loading {output_path}.")
            saved_paths[suffix] = output_path
            return pd.read_csv(output_path)
        return None

    def _save_df(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if df is None or df.empty:
            print(f"No rows generated for {suffix}; skipping save.")
            return pd.DataFrame()
        output_path = os.path.join(augmented_dir, f"{base_name}_{suffix}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {suffix} data to {output_path} (size: {len(df)})")
        saved_paths[suffix] = output_path
        return df

    para_df = _load_existing('rp')
    if para_df is None and num_paraphrase > 0:
        para_df = _save_df(augment_paraphrase(
            original_df, num_paraphrase), 'rp')
    elif para_df is None:
        para_df = pd.DataFrame()

    ss_df = _load_existing('ss')
    if ss_df is None and num_speaker_swap > 0:
        ss_df = _save_df(augment_speaker_swap(
            original_df, num_speaker_swap), 'ss')
    elif ss_df is None:
        ss_df = pd.DataFrame()

    synth_df = _load_existing('synthetic')
    if synth_df is None and num_synthetic > 0:
        synth_df = _save_df(
            generate_synthetic_data(
                original_df, num_synthetic, topics_with_weights=topic_weights),
            'synthetic'
        )
    elif synth_df is None:
        synth_df = pd.DataFrame()

    saved_paths['original'] = original_data_path
    print('Augmentation CSVs saved:', saved_paths)
    return saved_paths
