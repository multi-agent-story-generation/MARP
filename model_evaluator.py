import re
import numpy as np
import litellm
import time

from marp.standard_prompts import *
from dataset_loader import DatasetLoader


litellm.vertex_project = "multi-agent-411823"
litellm.vertex_location = "us-central1"


# Helper function
def get_response(model, message):
    """
    Query the LLM model with a message and return the response.
    """
    response = litellm.completion(
        model=model,
        messages=[{"content": message, "role": "user"}],
    )
    return response.choices[0].message.content


# Helper function
def extract_first_number(text):
        """
        Helper function to extract the first number encountered within a string.
        """
        match = re.search(r'\d+', text)
        if match:
            return int(match.group(0))
        else:
            return None


# Helper function
def compute_model_eval_acc(scores1, scores2, llmScores1, llmScores2):
        """
        Computes the model evaluation accuracy given the original and LLM scores.
        """
        assert len(scores1) == len(scores2) == len(llmScores1) == len(llmScores2)

        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        llmScores1 = np.array(llmScores1)
        llmScores2 = np.array(llmScores2)

        scores1 = scores1.reshape([len(scores1), -1])
        scores2 = scores2.reshape([len(scores2), -1])
        llmScores1 = llmScores1.reshape([len(llmScores1), -1])
        llmScores2 = llmScores2.reshape([len(llmScores2), -1])

        assert len(scores1[0]) == len(scores2[0]) == len(llmScores1[0]) == len(llmScores2[0])

        scores1 = scores1.sum(axis=1)
        scores2 = scores2.sum(axis=1)
        llmScores1 = llmScores1.sum(axis=1)
        llmScores2 = llmScores2.sum(axis=1)

        acc = 0
        
        for i in range(len(scores1)):
            if (scores1[i] - scores2[i]) * (llmScores1[i] - llmScores2[i]) > 0:
                acc += 1
            elif (scores1[i] - scores2[i]) == (llmScores1[i] - llmScores2[i]) == 0:
                acc += 1
        return acc / len(scores1)


class ModelEvaluator():
    def __init__(self, model, dataset_name, filepath, num_prompts_eval=3, 
                 num_categories=1, bidir_eval=False, eval_rounds=1):
        """
        num_prompts_eval: number of prompts to evaluate
        num_categories: number of scoring categories to evaluate

        bidir_eval: whether to evaluate both orders (directions) of presenting the two stories
        eval_rounds: number of rounds of evaluation for the same two stories

        num_rounds: number of rounds of evaluation for the same two stories (if bidi_eval is true, evaluations for both directions count together as one round).
        """
        self.model = model
        self.dataset_name = dataset_name
        self.filepath = filepath
        self.num_prompts_eval = num_prompts_eval
        self.num_categories = num_categories
        self.bidir_eval = bidir_eval
        self.eval_rounds = eval_rounds

        if dataset_name == "hanna":
            self.evaluate = self.evaluateHanna
            self.evaluateModels = self.evaluateModelsHanna
            self.num_all_prompts = 96
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
    def parse_scores(self, response, double_story=True, keyword="Story"):
        """
        Parses the scores from the evaluator's response.
        The response should be in the following format (the categories are just examples):
        Story1
        Relevance: 3
        Coherence: 4
        Empathy: 5
        Surprise: 2
        Engagement: 4
        Complexity: 3
        Story2
        Relevance: 3
        Coherence: 4
        Empathy: 5
        Surprise: 2
        Engagement: 4
        Complexity: 3
        """
        try:
            # Splitting the response by newlines
            lines = response.strip().split('\n')
            
            # Placeholder for scores
            llmScore1, llmScore2 = [], []

            # Variable to keep track of which story's scores we are currently parsing
            current_story = 1
        
            for line in lines:
                if (keyword is not None) and (keyword in line):
                    story_num = extract_first_number(line)
                    if story_num == 1:
                        current_story = 1
                    elif story_num == 2:
                        current_story = 2
                    else:
                        raise ValueError(f"Invalid story number in response:\n{response}")
                else:
                    score = extract_first_number(line)
                    if score is None:
                        continue
                    if current_story == 1:
                        llmScore1.append(score)
                    else:
                        llmScore2.append(score)

            if double_story:
                if len(llmScore1) != self.num_categories or len(llmScore2) != self.num_categories:
                    raise ValueError(f"Incorrect number of scoring categories for one or both stories.\nThe current response is:\n{response}")

                return llmScore1, llmScore2
            else:
                if len(llmScore1) != self.num_categories:
                    raise ValueError(f"Incorrect number of scoring categories for the story.\nThe current response is:\n{response}")
                return llmScore1
        
        except Exception as e:
            # Handling any potential errors
            raise ValueError(f"Error parsing scores: {e}")

    def evaluate_stories(self, premises, stories1, stories2):
        """
        Using an LLM, evaluates two lists of stories generated by two different entities based on given premises.

        premises: list of premises
        stories1: list of articles generated by entity1
        stories2: list of articles generated by entity2
        scores1: list of scores for stories1. Each score is a list of integers for each category.
        scores2: list of scores for stories2. Each score is a list of integers for each category.

        return: list of LLM scores for stories1, list of LLM scores for stories2
        """
        assert len(premises) == len(stories1) == len(stories2)
        
        llmScores1, llmScores2 = [], []
        
        for i in range(len(stories1)):
            premise = premises[i]
            story1 = stories1[i]
            story2 = stories2[i]

            prompt = HANNA_RATE_ESSAY_PROMPT_TEMPLATE.format(premise, story1, story2)

            repeat_query = True
            max_tries = 5
            while repeat_query and max_tries > 0:
                response = get_response(self.model, prompt)
                try:
                    llmScore1, llmScore2 = self.parse_scores(response, double_story=True, keyword="Story")
                    repeat_query = False
                except:
                    repeat_query = True
                    max_tries -= 1
                    if max_tries == 0:
                        llmScore1, llmScore2 = None, None

            if llmScore1 is None or llmScore2 is None:
                raise Exception(f"Error evaluating stories for premise {i+1}:\n\n{premise}")
            else:
                llmScores1.append(llmScore1)
                llmScores2.append(llmScore2)

        llmScores1 = np.array(llmScores1).sum(axis=1)
        llmScores2 = np.array(llmScores2).sum(axis=1)
        return llmScores1, llmScores2
    
    def evaluteSingleStory(self, premise, story):
        prompt = HANNA_RATE_SINGLE_ESSAY_PROMPT_TEMPLATE.format(premise, story)

        repeat_query = True
        max_tries = 5
        while repeat_query and max_tries > 0:
            response = get_response(self.model, prompt)
            try:
                llmScore = self.parse_scores(response, double_story=False, keyword=None)
                repeat_query = False
            except:
                repeat_query = True
                max_tries -= 1
                if max_tries == 0: 
                    llmScore = None

        if llmScore is None:
            raise Exception(f"Error evaluating stories for premise:\n\n{premise}")
        else:
            llmScore = sum(llmScore)
            return llmScore
            
    
    def evaluatePrefixHanna(self):
        NUM_TRAIN = self.num_all_prompts // 2
        assert self.num_prompts_eval <= NUM_TRAIN

        writers = ["Human", "BertGeneration", "CTRL", "GPT", "GPT-2 (tag)", "GPT-2", "RoBERTa", "XLNet", "Fusion", "HINT", "TD-VAE"]

        loader = DatasetLoader(self.dataset_name, self.filepath, writers)

        prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories = loader.processData()

        train_set, test_set = loader.splitTrainTest(prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories, NUM_TRAIN)

        return writers, train_set, test_set

    def evaluateHanna(self):
        writers, train_set, test_set = self.evaluatePrefixHanna()
        trainPrompt2Idx, trainIdx2Prompt, trainPrompt2Scores, trainPrompt2Stories = train_set
        testPrompt2Idx, testIdx2Prompt, testPrompt2Scores, testPrompt2Stories = test_set

        acc = 0
        acc_count = 0
        for i in range(len(writers)):
            writer1 = writers[i]
            for j in range(i+1, len(writers)):
                writer2 = writers[j]
                premises = []
                scores1 = []
                scores2 = []
                stories1 = []
                stories2 = []
                
                for prompt in list(trainPrompt2Idx.keys())[:self.num_prompts_eval]:
                    premises.append(prompt)
                    stories1.append(trainPrompt2Stories[prompt][writer1])
                    stories2.append(trainPrompt2Stories[prompt][writer2])
                    scores1.append(trainPrompt2Scores[prompt][writer1])
                    scores2.append(trainPrompt2Scores[prompt][writer2])
                
                llmScores1 = np.zeros(self.num_prompts_eval)
                llmScores2 = np.zeros(self.num_prompts_eval)
                for _ in range(self.eval_rounds):
                    tmp_llmScores1, tmp_llmScores2 = self.evaluate_stories(premises, stories1, stories2)
                    llmScores1 += tmp_llmScores1
                    llmScores2 += tmp_llmScores2

                    # Bidirectional evaluation
                    if self.bidir_eval:
                        tmp_llmScores2, tmp_llmScores1 = self.evaluate_stories(premises, stories2, stories1)
                        llmScores2 += tmp_llmScores2
                        llmScores1 += tmp_llmScores1

                tmp_acc = compute_model_eval_acc(scores1, scores2, llmScores1, llmScores2)
                acc += tmp_acc
                acc_count += 1
                print(f"Train Accuracy for {writer1} vs {writer2}: {tmp_acc}; cumulative accuracy: {acc / acc_count}")
                
        
        acc /= (len(writers) * (len(writers) - 1) / 2)
        print(f"\nOverall Train Accuracy: {acc}")
        assert acc_count == len(writers) * (len(writers) - 1) / 2

    @staticmethod
    def sort_model_by_score(overall_scores: dict) -> list:
        """
        overal_scores: a dictionary whose keys are model names (str) and values are the overall scores for each model.
        """
        # Sorting the dictionary by values (scores) in descending order
        sorted_scores = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)

        # sorted_scores is now a list of tuples sorted by the score
        return sorted_scores

    def evaluateModelsHanna(self):
        """
        Compare the overall story generation performance of different models (including possibly human) on a given number of prompts. The total human-evaluted scores for the stories generated by each model are averaged. The models are then ranked from the best to the worst based on the average scores.
        
        return: a list of (model, accuracy) tuples, sorted by accuracy from highest to lowest.
        """
        writers, train_set, test_set = self.evaluatePrefixHanna()

        trainPrompt2Idx, trainIdx2Prompt, trainPrompt2Scores, trainPrompt2Stories = train_set
        testPrompt2Idx, testIdx2Prompt, testPrompt2Scores, testPrompt2Stories = test_set
        
        human_overall_scores = {writer: 0.0 for writer in writers}

        # Collect total human-evaluated scores over selected prompts for each writer model
        for writer in writers:
            for prompt in list(trainPrompt2Idx.keys())[:self.num_prompts_eval]:
                human_overall_scores[writer] += trainPrompt2Scores[prompt][writer]


        # Next, we will collect total scores evaluated by the scorer model over selected prompts for each writer model.
                
        # model_scores is a dictionary of dictionaries, where model_scores[prompt][writer] is the total model-evaluted score for the story generated by writer with respect to prompt.
        model_scores = {}
        for prompt in list(trainPrompt2Idx.keys())[:self.num_prompts_eval]:
            model_scores[prompt] = {writer: 0.0 for writer in writers}
        
        # Collect total scores evaluated by the scorer model over all selected prompts for each writer model.
        model_overall_scores = {writer: 0.0 for writer in writers}

        for prompt in list(trainPrompt2Idx.keys())[:self.num_prompts_eval]:
            for writer in writers:
                story = trainPrompt2Stories[prompt][writer]
                model_scores[prompt][writer] = self.evaluteSingleStory(prompt, story)
                model_overall_scores[writer] += model_scores[prompt][writer]
                print(f"Model score for {writer} on prompt {trainPrompt2Idx[prompt]}: {model_scores[prompt][writer]}")

                # Hanlding the 60 quotas/min limit for gemini-pro
                if self.model == "gemini-pro":
                    time.sleep(0.3)

        # Compare writer models pair-wise to check whether the scorer model agrees with the human evaluation.
        cumu_acc = 0
        acc_count = 0
        for i in range(len(writers)):
            writer1 = writers[i]
            for j in range(i+1, len(writers)):
                writer2 = writers[j]

                acc = 0
                for prompt in list(trainPrompt2Idx.keys())[:self.num_prompts_eval]:
                    acc_count += 1

                    human_score1 = trainPrompt2Scores[prompt][writer1]
                    human_score2 = trainPrompt2Scores[prompt][writer2]
                    llm_score1 = model_scores[prompt][writer1]
                    llm_score2 = model_scores[prompt][writer2]

                    if (human_score1 - human_score2) * (llm_score1 - llm_score2) > 0:
                        acc += 1
                        cumu_acc += 1
                    elif (human_score1 - human_score2) == (llm_score1 - llm_score2) == 0:
                        acc += 1
                        cumu_acc += 1
                    
                print(f"Train Accuracy for {writer1} vs {writer2}: {acc / self.num_prompts_eval}; cumulative accuracy: {cumu_acc / acc_count}")
        
        assert acc_count == len(writers) * (len(writers) - 1) / 2 * self.num_prompts_eval
        print(f"\nOverall Train Accuracy: {cumu_acc / acc_count}\n\n")

        # Obtain the ranking of the writer models based on the human evaluation
        sorted_human_overall_scores = self.sort_model_by_score(human_overall_scores)
        sorted_model_overall_scores = self.sort_model_by_score(model_overall_scores)
        sorted_model_overall_scores = [(model, score*3) for model, score in 
                                       sorted_model_overall_scores]

        print("Writer model ranking based on human evaluation:")
        for i in range(len(sorted_human_overall_scores)):
            print(f"{i+1}. {sorted_human_overall_scores[i][0]}: {sorted_human_overall_scores[i][1]}")
        print("\n")

        print("Wrtier model ranking based on model evaluation:")
        for i in range(len(sorted_model_overall_scores)):
            print(f"{i+1}. {sorted_model_overall_scores[i][0]}: {sorted_model_overall_scores[i][1]}")