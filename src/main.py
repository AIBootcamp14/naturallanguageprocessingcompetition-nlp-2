# src/main.py
import os
import argparse
import torch
from pathlib import Path
from dotenv import load_dotenv
from src.data.augment_data import run_augmentation
from src.data.merge_data import merge_datasets
from src.train.train_llm import train_llm_with_qlora
from src.inference.inference_llm import run_inference


def main():
    parser = argparse.ArgumentParser(
        description="LLM Workflow")
    parser.add_argument('--skip-augmentation', action='store_true',
                        help="Skip the data augmentation step.")
    parser.add_argument('--skip-training', action='store_true',
                        help="Skip the LLM fine-tuning step.")
    parser.add_argument('--skip-inference', action='store_true',
                        help="Skip the final inference step.")
    parser.add_argument('--num-paraphrase', type=int, default=2500,
                        help='Number of paraphrase samples to generate (default: 2500).')
    parser.add_argument('--num-speaker-swap', type=int, default=2500,
                        help='Number of speaker-swap samples to generate (default: 2500).')
    parser.add_argument('--num-synthetic', type=int, default=1000,
                        help='Number of synthetic samples to generate (default: 1000).')
    parser.add_argument('--merge-base-path', type=str, default=None,
                        help='Base CSV path to merge with augmented data (defaults to clean train).')
    parser.add_argument('--debug-small', action='store_true',
                        help='Use small debug dataset for quick end-to-end checks (skips augmentation assumptions).')
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from the latest checkpoint in the output directory.")
    parser.add_argument('--preview-dev-samples', action='store_true',
                        help='Preview fixed dev samples after each evaluation step.')

    args = parser.parse_args()

    load_dotenv()

    # 1. 데이터 경로 설정
    if args.debug_small:
        clean_train_path = 'data/debug/train_small.csv'
        clean_dev_path = 'data/debug/dev_small.csv'
        clean_test_path = 'data/debug/test_small.csv'
        augmented_train_path = clean_train_path
    else:
        clean_train_path = 'data/clean/denoise/train.csv'
        clean_dev_path = 'data/clean/denoise/dev.csv'
        clean_test_path = 'data/clean/denoise/test.csv'
        augmented_train_path = 'data/augmented/train_augmented.csv'

    merge_base_path = args.merge_base_path or clean_train_path

    # 2. 증강 설정
    augmentation_config = {
        'original_data_path': clean_train_path,
        'dev_data_path': clean_dev_path,
        'augmented_data_path': augmented_train_path,
        'num_paraphrase': args.num_paraphrase,
        'num_speaker_swap': args.num_speaker_swap,
        'num_synthetic': args.num_synthetic,
        'merge_base_path': merge_base_path,
    }

    # 3. 모델 및 프롬프트 설정
    model_name = "SOLAR"
    default_model_id = "Upstage/SOLAR-10.7B-Instruct-v1.0"
    # default_model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    debug_model_id = os.getenv(
        'DEBUG_MODEL_ID', 'Upstage/SOLAR-10.7B-Instruct-v1.0')
    model_id = debug_model_id if args.debug_small else default_model_id
    use_fewshot = not args.debug_small
    prompt_suffix = "debug" if args.debug_small else (
        "fewshot" if use_fewshot else "zeroshot")
    model_directory = f"model/{model_name.lower()}_qlora_{prompt_suffix}"

    # 4. LLM 파인튜닝 및 추론 설정
    training_config = {
        'output_dir': model_directory,
        'num_train_epochs': 2 if args.debug_small else 3,
        'per_device_train_batch_size': 1 if args.debug_small else 1,
        'gradient_accumulation_steps': 1 if args.debug_small else 16,
        'learning_rate': 2e-4,
        'lr_scheduler_type': "cosine",
        'warmup_ratio': 0.1,
        'logging_strategy': "steps",
        'logging_steps': 10,
        'per_device_eval_batch_size': 1 if args.debug_small else 1,
        'eval_accumulation_steps': 1 if args.debug_small else 1,
        'evaluation_strategy': "epoch",
        'save_strategy': "epoch",
        'save_total_limit': 3,
        'gradient_checkpointing': True,
        'gradient_checkpointing_kwargs': {'use_reentrant': False},
        'optim': 'paged_adamw_8bit',
        'load_best_model_at_end': not args.debug_small,
        'metric_for_best_model': "eval_loss",
        'greater_is_better': False,
        'fp16': True, 'bf16': False,
        'report_to': [] if args.debug_small else "wandb",
        'resume_from_checkpoint': args.resume,
    }

    llm_config = {
        'model': {'id': model_id},
        'data': {
            'train_path': augmented_train_path,  # 증강 데이터를 기본으로 사용
            'dev_path': clean_dev_path
        },
        'prompt': {
            'use_fewshot': use_fewshot,
            'num_fewshot': 1,
            'few_shot_max_chars': 1024,
            'few_shot_disable_chars': 1100,
        },
        'lora': {'r': 16, 'alpha': 32, 'dropout': 0.05},
        'training': training_config,
        'max_seq_length': 2048 if args.debug_small else 4096,
        'show_dev_predictions': args.preview_dev_samples,
    }

    if args.debug_small:
        llm_config['prompt']['num_fewshot'] = 1
        llm_config['prompt']['use_fewshot'] = True

    # ======================= 파이프라인 시작 =======================

    augmented_target = Path(augmented_train_path)
    if not args.skip_augmentation and augmented_target.exists():
        print('\nAugmented data already exists. Skipping generation.')
    # Step 1: 데이터 증강
    if not args.skip_augmentation and not augmented_target.exists():
        print('\n' + '='*60)
        print('============ Starting Data Augmentation =============')
        print('='*60 + '\n')
        augmentation_opts = augmentation_config.copy()
        merge_base_path = augmentation_opts.pop(
            'merge_base_path', clean_train_path)
        saved_paths = run_augmentation(**augmentation_opts)

        merge_datasets(
            base_path=merge_base_path,
            augmented_paths=[saved_paths.get('rp'), saved_paths.get(
                'ss'), saved_paths.get('synthetic')],
            output_path=augmented_train_path,
            shuffle=True,
            random_state=42,
        )
    elif args.skip_augmentation:
        print('\nSkipping Step 1: Data Augmentation.')

    # Step 2: LLM 파인튜닝
    if not args.skip_training:
        print("\n" + "="*60)
        print("============ STEP 2: Starting LLM Fine-tuning ==============")
        print("="*60 + "\n")
        train_llm_with_qlora(llm_config)
    else:
        print("\nSkipping Step 2: LLM Fine-tuning.")

    # Step 3: 추론
    if not args.skip_inference:
        print("\n" + "="*60)
        print("============ STEP 3: Starting Inference ==============")
        print("="*60 + "\n")
        final_checkpoint_path = os.path.join(
            model_directory, "final_checkpoint")
        if not os.path.exists(final_checkpoint_path):
            raise FileNotFoundError(
                f"Trained model not found at {final_checkpoint_path}. Please run training first or check the path.")

        # GPU 사용 가능 여부 확인 후 CPU 추론 설정
        use_cpu = not torch.cuda.is_available() or torch.cuda.memory_allocated() > 0.8 * \
            torch.cuda.get_device_properties(0).total_memory

        run_inference(
            base_model_id=model_id,
            lora_adapter_path=final_checkpoint_path,
            test_data_path=clean_test_path,
            output_path='output/submission_llm_debug.csv' if args.debug_small else 'output/submission_llm.csv',
            batch_size=8 if use_cpu else 4,  # CPU에서는 더 큰 배치 사용
            few_shot_examples_path=augmented_train_path if use_fewshot else None,
            num_fewshot=llm_config['prompt'].get('num_fewshot', 1),
            few_shot_max_chars=llm_config['prompt'].get('few_shot_max_chars'),
            few_shot_disable_chars=llm_config['prompt'].get(
                'few_shot_disable_chars', 1100),
            use_cpu=use_cpu
        )
    else:
        print("\nSkipping Step 3: Inference.")

    print("\n================ LLM Workflow DONE ================")


if __name__ == '__main__':
    main()
