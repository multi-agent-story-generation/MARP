import csv


class DatasetLoader:
    """
    Loads a story/essay evaluation dataset, presented as a csv file. The first line of the CSV file are the column names. Then each subsequent line is a row of data.
    """
    def __init__(self, dataset_name, filepath, writers):
        self.filepath = filepath
        self.dataset_name = dataset_name
        self.writers = writers
        if dataset_name == "hanna":
            self.num_prompts = 96
            self.num_human_evaluators = 3
            self.num_categories = 6
            self.processData = self.processDataHanna
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
    @staticmethod
    def loadCSV(filepath):
        """
        Load a csv file. The first line of the CSV file are the column names.
        Then each subsequent line is a row of data.
        """
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            row_num = 0
            for row in reader:
                if row_num == 0:
                    column_names = row
                    data = []
                else:
                    data.append(row)
                row_num += 1

        return column_names, data

    def processDataHanna(self):
        """
        Load the dataset. The first line of the CSV file are the column names.
        Then each subsequent line is a row of data.
        """
        column_names, data = DatasetLoader.loadCSV(self.filepath)
        writers = self.writers

        idx2Prompt = {}
        prompt2Idx = {}
        prompt2Scores = {}

        # Helper dictionary to ensure each writer version of a story is evaluated {self.num_human_evaluators} times.
        prompt2AddCount = {}
        prompt2Stories = {}

        idx = 0
        for row in data:
            prompt = row[1].strip()
            if prompt not in prompt2Idx:
                prompt2Idx[prompt] = idx
                idx2Prompt[idx] = prompt
                idx += 1
                prompt2Scores[prompt] = {}
                prompt2AddCount[prompt] = {}
                prompt2Stories[prompt] = {}
                for writer in writers:
                    prompt2Scores[prompt][writer] = 0
                    prompt2AddCount[prompt][writer] = 0

            writer = row[4].strip()
            story = row[3].strip()

            relevance = int(row[5])
            coherence = int(row[6])
            empathy = int(row[7])
            surprise = int(row[8])
            engagement = int(row[9])
            complexity = int(row[10])
            score = relevance + coherence + empathy + surprise + engagement + complexity


            prompt2Scores[prompt][writer] += score
            prompt2AddCount[prompt][writer] += 1
            prompt2Stories[prompt][writer] = story
        
        for value in prompt2AddCount.values():
            for writer in writers:
                assert value[writer] == self.num_human_evaluators
        
        assert len(prompt2Idx) == len(idx2Prompt) == len(prompt2Scores) == len(prompt2AddCount) == self.num_prompts

        return prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories
    
    @staticmethod
    def splitTrainTest(prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories, num_train):
        """
        Splits the data into train and test sets.
        """
        trainPrompt2Idx = {}
        trainIdx2Prompt = {}
        trainPrompt2Scores = {}
        trainPrompt2Stories = {}

        testPrompt2Idx = {}
        testIdx2Prompt = {}
        testPrompt2Scores = {}
        testPrompt2Stories = {}

        for prompt in prompt2Idx:
            idx = prompt2Idx[prompt]
            if idx < num_train:
                trainPrompt2Idx[prompt] = idx
                trainIdx2Prompt[idx] = prompt
                trainPrompt2Scores[prompt] = prompt2Scores[prompt]
                trainPrompt2Stories[prompt] = prompt2Stories[prompt]
            else:
                testPrompt2Idx[prompt] = idx
                testIdx2Prompt[idx] = prompt
                testPrompt2Scores[prompt] = prompt2Scores[prompt]
                testPrompt2Stories[prompt] = prompt2Stories[prompt]

        assert len(trainPrompt2Idx) == len(trainIdx2Prompt) == len(trainPrompt2Scores) == len(trainPrompt2Stories) == num_train
        assert len(testPrompt2Idx) == len(testIdx2Prompt) == len(testPrompt2Scores) == len(testPrompt2Stories) == len(prompt2Idx) - num_train

        return [trainPrompt2Idx, trainIdx2Prompt, trainPrompt2Scores, trainPrompt2Stories], [testPrompt2Idx, testIdx2Prompt, testPrompt2Scores, testPrompt2Stories]