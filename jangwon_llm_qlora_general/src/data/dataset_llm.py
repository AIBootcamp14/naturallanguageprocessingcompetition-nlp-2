# src/data/dataset_llm.py

def create_prompt(row):

    dialogue = row['dialogue']
    summary = row['summary']

    prompt = f"""### 지시:
    당신은 주어진 대화를 자연스러운 한국어로 요약해야 합니다. 다음 원칙을 따르세요.
    1. 대화의 핵심 사건과 의사결정 또는 합의를 빠짐없이 전달합니다.
    2. 요약 길이는 원문 대비 약 20% 이내로 간결하게 유지합니다.
    3. 사람 이름, 조직명 등 중요한 명명된 개체는 원형 그대로 보존합니다.
    4. 관찰자의 시각에서 화자들의 의도와 행동을 중립적으로 설명합니다.
    5. 은어·약어를 피하고 공식적이고 명료한 문장으로 작성합니다.
    6. 원문에 등장한 화자 태그(`#Person1#`, `#Person2#`, `#Person3#` 등)가 요약에 필요하면 동일하게 사용합니다.

    ### 대화:
    {dialogue}

    ### 요약:
    {summary}"""

    return prompt


def create_prompt_with_fewshot(row, few_shot_samples):

    dialogue = row['dialogue']
    summary = row['summary']

    # Few-shot 예시 만들기
    example_texts = []
    for _, shot in few_shot_samples.iterrows():
        example_texts.append(f"""### 대화:
        {shot['dialogue']}

        ### 요약:
        {shot['summary']}""")

    example_section = "\n\n---\n\n".join(example_texts)

    # 최종 프롬프트 조합
    prompt = f"""### 지시:
    당신은 대화 요약 전문가입니다. 아래 지침과 예시를 참고하여 주어진 대화문을 요약하세요.

    [지침]
    1. 대화의 핵심 사건과 의사결정 또는 합의를 빠짐없이 전달합니다.
    2. 요약 길이는 원문 대비 약 20% 이내로 간결하게 유지합니다.
    3. 사람 이름, 조직명 등 중요한 명명된 개체는 원형 그대로 보존합니다.
    4. 관찰자의 시각에서 화자들의 의도와 행동을 중립적으로 설명합니다.
    5. 은어·약어를 피하고 공식적이고 명료한 문장으로 작성합니다.
    6. 원문에 등장한 화자 태그(`#Person1#`, `#Person2#`, `#Person3#` 등)가 요약에 필요하면 동일하게 사용합니다.

    ### 예시:
    {example_section}

    ---

    ### 대화:
    {dialogue}

    ### 요약:
    {summary}"""

    return prompt


def create_inference_prompt(row, few_shot_samples=None):
    """
    추론을 위한 프롬프트를 생성합니다. ### 요약: 까지만 포함됩니다.
    """
    dialogue = row['dialogue']

    # Few-shot 예시가 제공된 경우, 프롬프트에 포함
    if few_shot_samples is not None and not few_shot_samples.empty:
        example_texts = []
        for _, shot in few_shot_samples.iterrows():
            example_texts.append(f"""### 대화:
            {shot['dialogue']}

            ### 요약:
            {shot['summary']}""")

        example_section = "\n\n---\n\n".join(example_texts)

        prompt = f"""### 지시:
        당신은 대화 요약 전문가입니다. 아래 지침과 예시를 참고하여 주어진 대화문을 요약하세요.

        [지침]
        1. 대화의 핵심 사건과 의사결정 또는 합의를 빠짐없이 전달합니다.
        2. 요약 길이는 원문 대비 약 20% 이내로 간결하게 유지합니다.
        3. 사람 이름, 조직명 등 중요한 명명된 개체는 원형 그대로 보존합니다.
        4. 관찰자의 시각에서 화자들의 의도와 행동을 중립적으로 설명합니다.
        5. 은어·약어를 피하고 공식적이고 명료한 문장으로 작성합니다.
        6. 원문에 등장한 화자 태그(`#Person1#`, `#Person2#`, `#Person3#` 등)가 요약에 필요하면 동일하게 사용합니다.

        ### 예시:
        {example_section}

        ---

        ### 대화:
        {dialogue}

        ### 요약:
        """

    # Zero-shot (Few-shot 예시가 없는 경우)
    else:
        prompt = f"""### 지시:
        당신은 주어진 대화를 자연스러운 한국어로 요약해야 합니다. 다음 원칙을 따르세요.
        1. 대화의 핵심 사건과 의사결정 또는 합의를 빠짐없이 전달합니다.
        2. 요약 길이는 원문 대비 약 20% 이내로 간결하게 유지합니다.
        3. 사람 이름, 조직명 등 중요한 명명된 개체는 원형 그대로 보존합니다.
        4. 관찰자의 시각에서 화자들의 의도와 행동을 중립적으로 설명합니다.
        5. 은어·약어를 피하고 공식적이고 명료한 문장으로 작성합니다.
        6. 원문에 등장한 화자 태그(`#Person1#`, `#Person2#`, `#Person3#` 등)가 요약에 필요하면 동일하게 사용합니다.

        ### 대화:
        {dialogue}

        ### 요약:
        """

    return prompt
