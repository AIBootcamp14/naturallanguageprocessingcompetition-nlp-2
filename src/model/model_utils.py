from transformers import BartForConditionalGeneration, BartConfig, AutoTokenizer, AddedToken, AutoConfig,AutoModelForSeq2SeqLM,T5ForConditionalGeneration,T5Tokenizer


# 학습을 위한 tokenizer와 사전 학습된 모델을 불러옵니다.
def load_tokenizer_and_model_for_train(config,device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    # bart_config = BartConfig().from_pretrained(model_name)
    # bart_config.tie_word_embeddings = True
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # generate_model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'],config=bart_config)
    
    
    hf_config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    generate_model = AutoModelForSeq2SeqLM.from_pretrained(
                                                    model_name, 
                                                    config=hf_config)
    

    # 토크나이저 설정 확인 및 디버깅
    print(f"Tokenizer pad_token: {tokenizer.pad_token}")
    print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    print(f"Tokenizer eos_token: {tokenizer.eos_token}")
    print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")
    
    # t5 모델에 맞춰, pad/eos 동기화 추가 로직
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    generate_model.config.pad_token_id = tokenizer.pad_token_id
    generate_model.config.eos_token_id = tokenizer.eos_token_id
    
    if generate_model.config.decoder_start_token_id is None:
        generate_model.config.decoder_start_token_id = tokenizer.pad_token_id


    # special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    # tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.add_special_tokens({"additional_special_tokens": [
    "<topic>", "</topic>", "<dialogue>", "</dialogue>"]})
    
    basic_tokens = [
    "#Person1#", "#Person2#", "#Person3#", "#PhoneNumber#", "#Address#",
    "#PassportNumber#", "#Email#", "#Date#", "#Time#", "#Number#",
    "#Money#", "#Organization#", "#Location#",
    ]

    tokenizer.add_tokens([AddedToken(t, special=False) for t in basic_tokens])
    
    generate_model.resize_token_embeddings(len(tokenizer)) # 사전에 special token을 추가했으므로 재구성 해줍니다.
    generate_model.to(device)
    # print(generate_model.config)
    

    print('-'*10, 'Load tokenizer & model complete', '-'*10,)
    return generate_model , tokenizer



# 추론을 위한 tokenizer와 학습시킨 모델을 불러옵니다.
def load_tokenizer_and_model_for_test(config,device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)

    model_name = config['general']['model_name']
    ckt_path = config['inference']['ckt_path']
    print('-'*10, f'Model Name : {model_name}', '-'*10,)
    
    # tokenizer = AutoTokenizer.from_pretrained(ckt_path)
    # generate_model = BartForConditionalGeneration.from_pretrained(ckt_path)
    
    # T5 : psyche/KoT5-summarization
    # generate_model = T5ForConditionalGeneration.from_pretrained(ckt_path)
    # tokenizer = T5Tokenizer.from_pretrained(ckt_path)
    
    # T5 : lcw99/t5-base-korean-text-summary
    generate_model = AutoModelForSeq2SeqLM.from_pretrained(ckt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckt_path)
                                                    
    
    generate_model.to(device)
    generate_model.eval()
    
    print('-'*10, 'Load tokenizer & model complete', '-'*10,)

    return generate_model , tokenizer