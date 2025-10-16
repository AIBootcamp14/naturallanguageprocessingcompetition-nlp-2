
"""Local HyperCLOVA X text generation helper."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class LocalClovaX:
    """Load HyperCLOVA X locally and run chat-style generations."""

    def __init__(self, model_cfg: dict[str, Any], generation_cfg: dict[str, Any] | None):
        if "model_id" not in model_cfg:
            raise ValueError("model.model_id must be provided in the config")

        self.model_id = str(model_cfg["model_id"])
        self.quant = str(model_cfg.get("quant", "4bit")).lower()
        self.local_dir = model_cfg.get("local_dir") or None
        self.revision = model_cfg.get("revision") or None
        self.offline = bool(model_cfg.get("offline", False))
        self.explicit_device = model_cfg.get("device") or "auto"
        self.use_flash_attn = bool(model_cfg.get("use_flash_attn", False))
        self.token = (
            model_cfg.get("token")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
        )

        self.processor = None
        self.tokenizer = None
        self.model = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path: str | None = None

        self.generation_cfg = {
            "max_new_tokens": 180,
            "temperature": 0.2,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
            "do_sample": False,
        }
        if generation_cfg:
            self.generation_cfg.update(generation_cfg)

        # generation_cfg 딕셔너리를 인스턴스 속성으로 복사 (generate에서 사용)
        self.max_new_tokens = int(self.generation_cfg["max_new_tokens"])
        self.temperature = float(self.generation_cfg["temperature"])
        self.top_p = float(self.generation_cfg["top_p"])
        self.repetition_penalty = float(
            self.generation_cfg["repetition_penalty"])
        self.do_sample = bool(self.generation_cfg["do_sample"])

        self._load_model()

    # ---------------------------------------------------------------------
    # Resolution helpers
    # ---------------------------------------------------------------------
    def _resolve_cached_snapshot(self) -> Path | None:
        """Find a cached snapshot for the requested model_id if it exists."""
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        safe_name = self.model_id.replace("/", "--")
        candidates: list[Path] = []
        if cache_root.exists():
            for base in cache_root.glob(f"models--{safe_name}/snapshots/*"):
                if base.is_dir():
                    candidates.append(base)
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
        return None

    def _resolve_model_path(self) -> str:
        if self.local_dir:
            path = Path(self.local_dir).expanduser()
            if path.exists():
                logger.info("Using local HyperCLOVA X model from %s", path)
                return str(path)
            logger.warning(
                "Configured local_dir %s not found; falling back to cache/remote", path)

        cached = self._resolve_cached_snapshot()
        if cached:
            logger.info("Using cached HyperCLOVA X snapshot at %s", cached)
            return str(cached)

        if self.offline:
            raise FileNotFoundError(
                "HyperCLOVA X model not found locally. Set model.local_dir to the downloaded path "
                "or disable offline mode to fetch from Hugging Face."
            )
        logger.info(
            "No local snapshot found; will download from Hugging Face (%s)", self.model_id)
        return self.model_id

    # ---------------------------------------------------------------------
    # Model loading
    # ---------------------------------------------------------------------
    def _inject_token(self, kwargs: dict[str, Any]) -> None:
        if self.token:
            kwargs.setdefault("token", self.token)

    def _load_model(self) -> None:
        common_kwargs: dict[str, Any] = {
            "trust_remote_code": True, "local_files_only": self.offline}
        if self.revision:
            common_kwargs["revision"] = self.revision
        self._inject_token(common_kwargs)

        model_path = self._resolve_model_path()
        self.model_path = model_path

        # Load processor/tokenizer
        tokenizer = None
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(
                model_path, **common_kwargs)
            if hasattr(processor, "tokenizer") and getattr(processor, "tokenizer", None) is not None:
                tokenizer = processor.tokenizer
        except Exception as exc:
            processor = None
            logger.debug("AutoProcessor load failed: %s", exc)
        self.processor = processor

        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    use_fast=True,              # 핵심!
                    local_files_only=True,      # offline이면 True 유지
                    **{k: v for k, v in common_kwargs.items() if k != "local_files_only"}
                )
            except Exception as e:
                # 혹시 fast 실패시(거의 없지만) 마지막 폴백
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    use_fast=None,              # 라이브러리에게 결정 맡김
                    local_files_only=True
                )
        self.tokenizer = tokenizer

        # ✅ pad_token_id 안전하게 설정 (없을 경우 eos_token_id 사용)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # (추가) 패딩 방향: 배치 추론에서 이득이 있을 수 있음
        self.tokenizer.padding_side = "left"

        # TF32 허용 (잔여 FP32 연산 가속)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 토크나이저 로드 직후 pad 토큰 세팅 (이미 넣었으면 유지)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # 배치 패딩 효율
        self.tokenizer.padding_side = "left"

        # ---- 여기서부터 모델을 '단 한 번' 로드 ----
        model_kwargs = {"trust_remote_code": True,
                        "attn_implementation": "flash_attention_2"}
        if self.revision:
            model_kwargs["revision"] = self.revision
        if self.offline:
            model_kwargs["local_files_only"] = True
        self._inject_token(model_kwargs)

        if torch.cuda.is_available():
            if self.quant in {"fp16", "float16"}:
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = torch.float16
            elif self.quant == "8bit":
                model_kwargs["device_map"] = "auto"
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True)
            elif self.quant == "4bit":
                model_kwargs["device_map"] = "auto"
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = None  # CPU

        # device override(선택)
        device_pref = str(self.explicit_device).lower(
        ) if self.explicit_device else "auto"
        if device_pref != "auto":
            if device_pref == "cpu":
                model_kwargs["device_map"] = None
            elif device_pref.startswith("cuda") and torch.cuda.is_available():
                model_kwargs["device_map"] = device_pref

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, **model_kwargs)
        self.model.eval()
        self.model.config.use_cache = True

        # device 객체 저장
        if model_kwargs.get("device_map") in (None, "cpu") or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.model.to(self.device)
        else:
            if isinstance(model_kwargs.get("device_map"), str) and model_kwargs["device_map"].startswith("cuda"):
                self.device = torch.device(model_kwargs["device_map"])
            else:
                self.device = torch.device("cuda")

    # ---------------------------------------------------------------------
    # Inference helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _normalize_messages_for_text_model(messages):
        norm = []
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, list):
                # 멀티모달/세그먼트형을 텍스트로 변환
                parts = []
                for p in c:
                    if isinstance(p, dict):
                        # VLM 스타일: {"type":"text","text":"..."} 만 취함
                        if p.get("type") == "text" and "text" in p:
                            parts.append(str(p["text"]))
                    else:
                        # 단순 리스트 ["문장1","문장2"] 같은 경우
                        parts.append(str(p))
                c = "\n".join(parts)
            else:
                c = str(c)
            norm.append({"role": m.get("role", "user"), "content": c})
        return norm

    def _apply_chat_template(self, messages):
        # 1) 메시지 정규화 (list content -> string)
        msgs = self._normalize_messages_for_text_model(messages)

        # 2) 템플릿을 먼저 '문자열'로 렌더링
        prompt = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,             # <-- 문자열로 받기 (중요)
            add_generation_prompt=True,
        )

        # 3) 문자열을 토크나이즈해서 dict(=BatchEncoding) 얻기
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,              # 배치 1이면 False가 깔끔
            truncation=False,           # 경고 없애기; 길면 max_new_tokens로 제어
        )
        # 디바이스로 이동
        return {k: v.to(self.model.device) for k, v in enc.items()}

    def render_prompt_as_text(self, messages) -> str:
        """messages(list[dict]) -> chat template를 문자열로 렌더링."""
        msgs = self._normalize_messages_for_text_model(messages)
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def tokenize_prompts(self, prompts: list[str]):
        """여러 문자열 프롬프트를 한 번에 토크나이즈(padding 포함)."""
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,      # 배치 패딩
            truncation=False,  # 길이 제한은 max_new_tokens로 제어
        )
        return {k: v.to(self.model.device, non_blocking=True) for k, v in enc.items()}

    @torch.inference_mode()
    def generate_batch(self, messages_list: list[list[dict]]):
        """배치로 generate. 각 샘플의 생성 텍스트만 리스트로 반환."""
        prompts = [self.render_prompt_as_text(msgs) for msgs in messages_list]
        enc = self.tokenize_prompts(prompts)

        # (추가) 샘플링 안 쓰면 temperature/top_p 제거
        temp = self.temperature if self.do_sample else None
        top_p = self.top_p if self.do_sample else None

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            temperature=temp,
            top_p=top_p,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        # (변경) fp16 autocast로 FA2 안전 가속
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.device.type == "cuda")):
            outs = self.model.generate(**enc, **gen_kwargs)

        # (유지) 입력 길이만큼 잘라서 '생성 텍스트만' 디코딩 → 프롬프트 유출 방지
        input_lens = enc["attention_mask"].sum(dim=1)
        texts = []
        for i in range(outs.size(0)):
            gen_tok = outs[i, int(input_lens[i].item()):]
            texts.append(self.tokenizer.decode(
                gen_tok, skip_special_tokens=True).strip())
        return texts

    def generate(self, messages):
        """단일 샘플용(기존 호출 호환)."""
        text_list = self.generate_batch([messages])
        return text_list[0] if text_list else ""
