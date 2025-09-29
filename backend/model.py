# model.py
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
import torch
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_device():
    if torch.cuda.is_available():
        # choose gpu 0 by default
        return "cuda:0"
    return "cpu"


def load_phi3_model(model_name: str = config.MODEL_NAME):
    device = get_device()
    logger.info("Loading model %s on device=%s", model_name, device)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.exception("Failed to load model %s: %s", model_name, e)
        raise

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        repetition_penalty=config.REPETITION_PENALTY,
        device=0 if "cuda" in device else -1,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return model, tokenizer, llm
