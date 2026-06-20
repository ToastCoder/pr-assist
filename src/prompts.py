# pr-assist
# prompts.py

REVIEW_INSTRUCTION = (
    "Review the following pull request and provide detailed feedback."
)


def format_pr_prompt(title: str, description: str, diff: str) -> str:
    """Formats pull request information into a structured prompt."""
    return (
        f"{REVIEW_INSTRUCTION}\n\n"
        f"PR Title:\n{title}\n\n"
        f"PR Description:\n{description}\n\n"
        f"Code Changes:\n\n{diff}"
    )


def sample_review_prompt() -> str:
    """Returns a small example prompt for smoke tests."""
    return format_pr_prompt(
        title="Add JWT authentication middleware and protect user endpoints",
        description=(
            "This PR introduces JWT authentication for API endpoints.\n"
            "Users must now provide a valid JWT token to access protected routes."
        ),
        diff=(
            '+ const jwt = require("jsonwebtoken");\n'
            "+\n"
            "+ function authenticate(req, res, next) {\n"
            "+   const token = req.headers.authorization;\n"
            "+\n"
            '+   if (!token) {\n'
            '+     return res.status(401).json({ error: "Unauthorized" });\n'
            "+   }\n"
            "+\n"
            "+   const decoded = jwt.verify(token, process.env.JWT_SECRET);\n"
            "+   req.user = decoded;\n"
            "+   next();\n"
            "+ }\n"
            "+\n"
            '+ router.get("/users", authenticate, getUsers);'
        ),
    )
