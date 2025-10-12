# model.py
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_phi4_model(model_name: str = config.MODEL_NAME):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info(f"Loaded model '{model_name}' on {next(model.parameters()).device}, dtype={model.dtype}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Phi 3 needs padding side as left
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    except Exception as e:
        logger.exception("Failed to load model %s: %s", model_name, e)
        raise
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=config.DO_SAMPLE,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        repetition_penalty=config.REPETITION_PENALTY,
        max_new_tokens=config.MAX_NEW_TOKENS,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return model, tokenizer, llm
