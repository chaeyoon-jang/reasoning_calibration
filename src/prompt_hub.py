########################## Reasoning Prompt Templates ##########################
BASE_REASONING = (
    "This is a conversation between **User** and **Assistant**.\n"
    + "The User asks a question, and the Assistant provides a solution.\n"
    + "Before answering, the Assistant reasons through the problem step-by-step.\n"
    + "The reasoning is enclosed within `<think> ... </think>`, and the final answer within `<answer> ... </answer>`.\n\n"
    + "Example:\n"
    + "{question}\n"
    + "<think>{step-by-step reasoning}</think>\n"
    + "<answer>{final answer}</answer>\n\n"
    + "Now, respond to the following using the **exact same format**:\n"
    + "<question>\n"
)

PREFIX_REASONING = (
    "This is a conversation between **User** and **Assistant**.\n"
    + "The User asks a question, and the Assistant provides a thoughtful, reasoned answer.\n\n"
    + "Before answering, the Assistant first reasons through the problem step-by-step.\n"
    + "The reasoning is enclosed in `<think> ... </think>` tags.\n"
    + "The final answer is enclosed in `<answer> ... </answer>` tags.\n"
    + "A confidence score is then provided in `<confidence> ... </confidence>` tags,\n"
    + "representing the Assistant’s certainty as a **continuous value between 0 and 100**.\n\n"
    + "**Example:**\n"
    + "{question}\n"
    + "<think>{step-by-step reasoning}</think>\n"
    + "<answer>{final answer}</answer>\n"
    + "<confidence>{confidence}</confidence>\n\n"
    + "Now, answer the following in **exactly** the same format:\n"
    + "<question>\n"
)

SUFFIX_CONFIDENCE = (
    "\nPlease respond with a score from 0 to 100 in `<confidence> </confidence>` tags. How confident are you in your previous answer? "
)

PREFIX_REASONING_OUT = (
    "This is a conversation between **User** and **Assistant**.\n"
    + "The User asks a question, and the Assistant provides a thoughtful, reasoned answer.\n\n"
    + "The Assistant first reasons step‑by‑step to find the answer. "
    + "This reasoning is enclosed in <think> ... </think> tags.\n"
    + "Next, the Assistant reflects on **its own confidence** in that reasoning, "
    + "enclosed in <confidence_think> ... </confidence_think> tags.\n"
    + "Then, a final answer is provided in <answer> ... </answer> tags.\n"
    + "Finally, the Assistant outputs a scalar confidence score in <confidence> ... </confidence> tags, "
    + "representing certainty on a scale from 0 to 100.\n\n"
    + "**Example:**\n"
    + "{question}\n"
    + "<think>{reasoning to solve the problem}</think>\n"
    + "<confidence_think>{reasoning about confidence}</confidence_think>\n"
    + "<answer>{final answer}</answer>\n"
    + "<confidence>{confidence}</confidence>\n\n"
    + "Now, answer the following in **exactly** the same format:\n"
    + "<question>\n"
)

SUFFIX_CONFIDENCE_OUT = (
    "\nFirst, reflect on your certainty inside <confidence_think> ... </confidence_think> tags, "
    +"then provide a single score from 0 to 100 inside <confidence> ... </confidence> tags.\n"
    +"How confident are you in your previous answer? "
)
################################################################################

####################### No Reasoning Prompt Templates ##########################
BASE_NO_REASONING = (
    "A conversation between **User** and **Assistant**. The User asks a question.\n\n"
    "The Assistant must reply with:\n"
    "- The final answer only\n"
    "- The answer must be enclosed within `<answer> ... </answer>` tags\n\n"
    "Now answer the following question in the exact same format:\n"
    "<question>\n"
)

PREFIX_NO_REASONING = (
    "A conversation between User and Assistant. The User asks a question.\n\n"
    + "The Assistant must respond with:\n"
    + "- Only the final answer\n"
    + "- No explanation, no reasoning, no steps\n"
    + "- The answer must be enclosed **exactly** within `<answer> </answer>` tags\n"
    + "- The assistant provides a confidence score enclosed within `<confidence> </confidence>` tags, where the confidence is a continuous value between 0 and 100, representing the assistant's certainty.\n"
    + "- Do not output anything else. Do not add any sentences.\n\n"
    + "Now answer the following in the exact same format:\n"
    + "<question>\n" 
)
#################################################################################

confidence_prompts = {
    'base': BASE_REASONING,
    'suffix': SUFFIX_CONFIDENCE,
    'prefix': PREFIX_REASONING,
    'prefix_out': PREFIX_REASONING_OUT
}

confidence_prompts_no_reasoning = {
    'base': BASE_NO_REASONING,
    'suffix': SUFFIX_CONFIDENCE,
    'prefix': PREFIX_NO_REASONING
}


PARSING_PROMPT  = """Instruction:
We have a user's question and a model's generated response:

Your task:
1. Carefully read the question and the generated response in **Example 6 only**.
2. Extract the final answer based on the following rules:
    - If the response contains a number (with or without units), **extract only the numeric value**.
    - If the response is purely textual (no numbers), **extract the exact string as it appears**.
3. Use the following output format:
    - **Model's Final Answer is:** [Your extracted answer]

Rules:
- Only process **Example 6** for extraction. Ignore all other examples.
- Do not include units, symbols, or extra text when extracting numbers.
- Provide the answer strictly in the requested format without additional explanations.

### Examples

#### Example 1:
- Model's Generated Response: It takes about 160 minutes.
- **Model's Final Answer is:** 160

#### Example 2:
- Model's Generated Response: The nearest star is approximately 4.24 light years away.
- **Model's Final Answer is:** 4.24

#### Example 3:
- Model's Generated Response: The tallest mountain is Mount Everest.
- **Model's Final Answer is:** Mount Everest

#### Example 4:
- Model's Generated Response: It weighs 5 kg.
- **Model's Final Answer is:** 5

#### Example 5:
- Model's Generated Response: 81 + 221 - 24 = 278.
- **Model's Final Answer is:** 278

#### Example 6:
- Model's Generated Response: {answer_text}"""