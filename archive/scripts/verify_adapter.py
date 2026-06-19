# pr-assist
# verify_adapter.py

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "toastcoder/pr-review-qwen-lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER
)

prompt = """
Review the following pull request and provide detailed feedback.

PR Title:
Add JWT authentication middleware and protect user endpoints

PR Description:
This PR introduces JWT authentication for API endpoints.
Users must now provide a valid JWT token to access protected routes.

Code Changes:

+ const jwt = require("jsonwebtoken");
+
+ function authenticate(req, res, next) {
+   const token = req.headers.authorization;
+
+   if (!token) {
+     return res.status(401).json({ error: "Unauthorized" });
+   }
+
+   const decoded = jwt.verify(token, process.env.JWT_SECRET);
+   req.user = decoded;
+   next();
+ }
+
+ router.get("/users", authenticate, getUsers);
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("Response: \n", response)