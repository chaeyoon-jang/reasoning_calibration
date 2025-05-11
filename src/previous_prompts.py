USER_PROMPT_FOR_SYSTEM_CONFIDENCE = (
    "A conversation between the User and the Assistant. "
    + "The User asks a question, and the Assistant provides a solution.\n"
    + "The Assistant first reasons through the problem before providing the answer and a confidence score.\n"
    + "The reasoning process is enclosed within `<think> </think>` tags.\n"
    + "The final answer (without full sentences) is enclosed within `<answer> </answer>` tags.\n"
    + "The Assistant provides a confidence score enclosed within `<confidence> </confidence>` tags, "
    + "where the confidence is expressed using filled hearts (♥), with each filled heart representing 10% confidence. "
    + "A total of 10 hearts should be used, with empty hearts (♡) indicating the missing percentage.\n"
    + "For example, if the confidence level is 50%, the output should be: '<confidence>♥♥♥♥♥♡♡♡♡♡</confidence>'.\n\n"
    + "Q: <question>\n"
    + "A: Let's think step by step."
)


USER_PROMPT_FOR_SYSTEM_CONFIDENCE_NO_REASONING = (
    "A conversation between the User and the Assistant. "
    "The User asks a question, and the Assistant provides an immediate solution without any explanation.\n"
    "The final answer (without full sentences) is enclosed within `<answer> </answer>` tags.\n"
    "The Assistant provides a confidence score enclosed within `<confidence> </confidence>` tags, "
    "where the confidence is expressed using filled hearts (♥), with each filled heart representing 10% confidence. "
    "A total of 10 hearts should be used, with empty hearts (♡) indicating the missing percentage.\n"
    "For example, if the confidence level is 50%, the output should be: '<confidence>♥♥♥♥♥♡♡♡♡♡</confidence>'.\n\n"
    "Q: <question>\n"
    "A: "
)



USER_PROMPT_FOR_BINARY_CONFIDENCE = (
    "A conversation between User and Assistant."
    + "The user asks a question, and the Assistant solves it.\n"
    + "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer and a binary confidence score.\n"
    + "The reasoning process and answer are enclosed within `<think> </think>` and `<answer> </answer>` tags, respectively.\n"
    + "Additionally, the assistant provides a confidence score enclosed within `<confidence> </confidence>` tags, "
    + "where the confidence is either 0 (not confident) or 1 (confident), representing the assistant's certainty.\n"
    + "Q: <question>\n"
    + "A: Let's think step by step."
)


USER_PROMPT_FOR_DISCRETE_CONFIDENCE = (
    "A conversation between User and Assistant."
    + "The user asks a question, and the Assistant solves it.\n"
    + "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer and a discrete confidence score.\n"
    + "The reasoning process and answer are enclosed within `<think> </think>` and `<answer> </answer>` tags, respectively.\n"
    + "Additionally, the assistant provides a confidence score enclosed within `<confidence> </confidence>` tags, "
    + "where the confidence can take one of the following values: {0, 1, 2, 3, 4}, representing the assistant's level of certainty.\n"
    + "Q: <question>"
    + "A: Let's think step by step."
)


BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant."
)


BASE_USER_PROMPT = (
    "Q: <question>\n"
)