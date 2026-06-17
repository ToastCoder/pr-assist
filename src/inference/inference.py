from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from peft import PeftModel

import torch

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "models/pr-assist-qwen/final"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype="auto"
    )

    model = PeftModel.from_pretrained(
        model,
        LORA_PATH
    )

    model.eval()

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt,
):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


def read_multiline_input():
    print("\nPaste prompt below.")
    print("Type END on a new line when finished.\n")

    lines = []

    while True:
        line = input()

        if line.strip() == "END":
            break

        lines.append(line)

    return "\n".join(lines)


def main():
    print("Loading model...")

    model, tokenizer = load_model()

    print(type(model))
    print("Model ready")

    while True:
        prompt = read_multiline_input()

        if prompt.lower() in [
            "quit",
            "exit",
        ]:
            break

        print("\nGenerating...\n")

        response = generate_response(
            model,
            tokenizer,
            prompt,
        )

        print("Assistant:\n")
        print(response)
        print()


if __name__ == "__main__":
    main()