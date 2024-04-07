# rate prompts
GENERATE_ESSAY_PROMPT_TEMPLATE = "Based on premise: \"{}\" generate story containing several scenes, use scene1:, scene2:, ... to represent."

RATE_ESSAY_PROMPT_TEMPLATE="Based on 1. Interesting. Interesting to the reader. 2. Coherent. Plot-coherent. 3. Relevant. Faithful to the initial premise. 4. Humanlike. Judged to be human-written.4 dimensions evaluate following 2 stories, the score is from 0 to 100, higher score means better.\nThe initial premise of story is \"{}\"\nStory 1: {}\n Story 2: {}."

HANNA_RATE_ESSAY_PROMPT_TEMPLATE="Based on the following six categories: 1. Relevance. 2. Coherence. 3. Empathy. 4. Surprise. 5. Engagement. 6. Complexity, evaluate the following two stories by assigning an integer score (from 1 to 5) to each category. Higher score means better.\nThe initial premise of story is: {}\nStory1: {}\n Story2: {}.\n\nIn your response, please use the following example format strictly and no need for any extra explanations: \nStory1\nRelevance: \nCoherence: \nEmpathy: \nSurprise: \nEngagement: \nComplexity: \nStory2\nRelevance: \nCoherence: \nEmpathy: \nSurprise: \nEngagement: \nComplexity: \n"

HANNA_RATE_SINGLE_ESSAY_PROMPT_TEMPLATE="Based on the following six categories: 1. Relevance. 2. Coherence. 3. Empathy. 4. Surprise. 5. Engagement. 6. Complexity, evaluate the following story by assigning an integer score (from 1 to 5) to each category. Higher score means better.\nThe initial premise of story is: {}\nStory: {}\n\nIn your response, please use the following example format strictly and no need for any extra explanations:\nRelevance: \nCoherence: \nEmpathy: \nSurprise: \nEngagement: \nComplexity: \n"

# HANNA_RATE_ESSAY_PROMPT_TEMPLATE="Based on the following four categories: 1. Interesting, 2. Coherent, 3. Relevant, 4. Humanlike, evaluate the following two stories by assigning an integer score (from 1 to 5) to each category. Higher score means better.\nThe initial premise of story is: {}\n\nStory1: {}\n\n Story2: {}.\n\nIn your response, please use the following example format strictly and no need for any extra explanations: \nStory1\nInteresting: 3\nCoherent: 4\nRelevant: 5\nHumanlike: 2\nStory2\nInteresting: 1\nCoherent: 5\nRelevant: 5\nHumanlike: 3\n"

# HANNA_RATE_ESSAY_PROMPT_TEMPLATE="Evaluate the following two stories by assigning an integer score (from 1 to 30) to each. Higher score means better.\nThe initial premise of story is: {}\n\nStory1: {}\n\n Story2: {}.\n\nIn your response, please use the following example format strictly and no need for any extra explanations: \nStory1\nScore: 19\nStory2\nScore: 22\n"

# story prompts
QUARREL_PREMISE = "You will collaborate to create a story. The general setting: A Quarrel between two good friends about Iron Man."
IBRUSIA_PREMISE = "You will collaborate to create a story. The general setting: The state of Ibrusia is coming to a desperate and dangerous situation as the Hosso Union approaches its capital, Zaragoza."
ECONOMY_PREMISE = "You will collaborate to create a story. The general setting: The state of Gurata is coming to a huge economic recession. People are in panic and streets are in turmoil."

# player prompts
CONTROLLER_PROMPT = "You are the scene coordinator of a story. Your job is to select the next one actor that should act. You will be given several rounds of previous conversation in the play. If you think a player should be on stage next, print the player's name. For example: '### Next up: Amy' or '### Next up: Sheldon'. If you think the scene should end, then print '### Next up: END'. You should try not to repeat choosing the same player."
GLOBAL_DESIGNER_PROMPT = "You are the designer of a story. Your job is to design a global setting for this story. The topic of your story is '{}'. Please compose a setting (sufficient but not verbose) about the background, and design the characters in this setting. For example, a valid output is: 'The global scene is at the Yale University Bass Library. * Jane: A 2nd year computer science undergraduate student. * George: A first-year computer science master student. * Jake: A professor in computer science specializing in AI.'"
DESIGNER_PROMPT = "You are the designer of a story. Previously, you have written a global setting, now your job is to design the setting of the current scene and allocate the players that should be present in this scene. Be sure to pick a player from the list given below. Your output should follow the format of <setting>\n### Next up: <character1>, <character2>, ... For example: 'The current scene is set in a communication room. ### Next up: Brook, Elliot' 'The current scene is set in the grand drawing room. ### Next up: Jane, George'"
WRITER_PROMPT = "Given the previous scene of the gameplay, your role is to convert the conversations in this gameplay into one chapter of a story. Note that your output might be directly concatenated to previous outputs or future outputs, so do not structure it as a complete story, but rather a fragment of one. Here are some instructions about the language: 1. Don't make the story read like a summary. 2. When crafting the story, please respect the contents of the provided conversations and in the mean time make the story coherent. 3. You can add some embellishments, but don't be verbose. 4. Try to keep as much conversation as possible. 5. DON'T say you are a writer."
ENV_MANAGER_PROMPT = "You are reading the script a popular play. In order to facilitate future reading, your job is to conclude each act of a player into a sentence. Only include critical information such as the actions of the player and their impacts, the emotion of the player, the change in the player's opinion or plan, etc. Your input Your output should follow the format of: ### Summary: <your summary> Here are some examples you've previously written, follow this pattern and keep up the good work: The act you summarize: Seated at an oak table laden with books and parchment, Brandon is the image of scholarly dedication, with a quill tucked behind one ear. The warm glow of a candle illuminates his focused expression, as he writes furiously, occasionally stopping to consult a large, leather-bound tome. ### Summary: Brandon is composing a large tome at an oak table by a candle. He is dedicated."
PLAYER_PROMPT = "You'll be given what you and other people have done previously. Please speak and act according to it. Try not to repeat what you did before."
READER_PROMPT = "You are the reader of a story. Your job is to read the story and provide feedback on the story. "

# action prompts
DESIGNER_ACTION_PROMPT = "Now please design Scene {} and make sure this scene setting is completely different from previous scene settings. Be sure to pick a player from the list given below. Your output should follow the format of <setting>\n### Next up: <character1>, <character2>, ... For example:\n'The current scene is set in a communication room.\n### Next up: Brook, Elliot'\n'The current scene is set in the grand drawing room. ### Next up: Jane, George'"
GLOBAL_DESIGNER_ACTION_PROMPT = "Now please design."
SUMMARIZER_ACTION_PROMPT = "Now please summarize."
CONTROLLER_ACTION_PROMPT = "Now please select a player."
ROLE_INIT_ACTION_PROMPT = "Now plan your actions in this play. Be careful and wise. This is your only chance to plan your actions, so be considerate and elaborate. Your output should follow the format of: '### My plan:\n <explanation and plan>'"
ROLE_ACT_ACTION_PROMPT = "Now you can speak and act. Please try to limit your speak and act content to be fewer than 4 sentences. Try not to repeat your words and actions in previous scene."
READER_ACTION_PROMPT = "Now please read the story and provide feedback on the story. You should evaluate the story based on the following four dimensions: 1. Interesting. Interesting to the reader. 2. Coherent. Plot-coherent. 3. Relevant. Faithful to the initial premise. 4. Humanlike. Judged to be human-written. You should provide a score from 0 to 100 for each dimension. Higher score means better. You should also provide a brief explanation for each score. For example: 'Interesting: 90. The story is very interesting and engaging. Coherent: 80. The plot is coherent and easy to follow. Relevant: 70. The story is relevant to the initial premise. Humanlike: 60. The story is human-like and well-written.'"