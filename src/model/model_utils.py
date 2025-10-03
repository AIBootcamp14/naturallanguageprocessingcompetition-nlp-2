from transformers import BartForConditionalGeneration, BartConfig, AutoTokenizer, AddedToken


# 학습을 위한 tokenizer와 사전 학습된 모델을 불러옵니다.
def load_tokenizer_and_model_for_train(config,device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    bart_config = BartConfig().from_pretrained(model_name)
    bart_config.tie_word_embeddings = True
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'],config=bart_config)

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
    tokenizer = AutoTokenizer.from_pretrained(ckt_path)

    generate_model = BartForConditionalGeneration.from_pretrained(ckt_path)
    generate_model.to(device)
    
    print('-'*10, 'Load tokenizer & model complete', '-'*10,)

    return generate_model , tokenizer