def format_sample(sample):
    return f"""
<|user|>
{sample["instruction"]}

<|assistant|>
{sample["output"]}
"""