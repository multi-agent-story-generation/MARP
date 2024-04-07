import litellm
from pathlib import Path
from tqdm import tqdm

from marp.standard_prompts import *
from model_evaluator import ModelEvaluator


def get_response(model, message):
    """
    Query the LLM model with a message and return the response.
    """
    response = litellm.completion(
        model=model,
        messages=[{"content": message, "role": "user"}],
    )
    return response.choices[0].message.content


def generate_baseline(baseline_model, premise, baseline_save_path):
    baseline = get_response(baseline_model, GENERATE_ESSAY_PROMPT_TEMPLATE.format(premise))
    with open(baseline_save_path, 'w+') as f:
        f.write(baseline)


def evaluate(essay_path, evaluator_model, premise, num_trials, baseline_path):
    with open(essay_path, 'r') as f:
        ours = f.read()
    with open(baseline_path, 'r') as f:
        baseline = f.read()
    evaluations = []
    for _ in tqdm(range(num_trials)):
        prompt = RATE_ESSAY_PROMPT_TEMPLATE.format(premise, ours, baseline)
        # if ('gpt-3.5' in evaluator_model or 'lama' in evaluator_model) and len(prompt) > 4096:
        #     print(f'truncating to 4096 tokens from {len(prompt)} tokens for {evaluator_model}')
        #     prompt = prompt[:4096]
        #     print(prompt)
        evaluation = get_response(evaluator_model, prompt)
        evaluations.append(evaluation)
    return evaluations


# baseline_path = 'gemini-ibrusia.txt'
# # generate_baseline('gemini-pro', IBRUSIA_PREMISE, baseline_path)
# for evaluator_model in ['gpt-3.5-turbo-16k', 'gemini-pro', 'anyscale/mistralai/Mistral-7B-Instruct-v0.1']:
#     print(f'Evaluating with {evaluator_model}...')
#     evaluations = evaluate(
#         essay_path='storys/mid_ibrusia.txt',
#         evaluator_model=evaluator_model,
#         premise=IBRUSIA_PREMISE,
#         num_trials=5,
#         baseline_path=baseline_path,
#     )
#     output_path = Path(f'evaluations/{evaluator_model}.txt')
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(output_path, 'w+') as f:
#         f.write(f'********{evaluator_model}********\n')
#         f.write('\n\n\n********************************\n\n\n'.join(evaluations))


if __name__ == '__main__':
    evaluator = ModelEvaluator('gemini-pro', 'hanna', 'hanna/hanna_stories_annotations.csv', 
                               num_prompts_eval=48, num_categories=6, bidir_eval=False, eval_rounds=1)

    # evaluator = ModelEvaluator('gpt-4-1106-preview', 'hanna', 'hanna/hanna_stories_annotations.csv', num_prompts_eval=48, num_categories=6, bidir_eval=False, eval_rounds=1)

    # evaluator = ModelEvaluator('gpt-3.5-turbo', 'hanna', 'hanna/hanna_stories_annotations.csv', 
    #                            num_prompts_eval=48, num_categories=6, bidir_eval=False, eval_rounds=1)

    # evaluator = ModelEvaluator('anyscale/mistralai/Mistral-7B-Instruct-v0.1', 'hanna', 
    #                            'hanna/hanna_stories_annotations.csv', num_prompts_eval=2, num_categories=6, bidir_eval=True, eval_rounds=1)

    evaluator.evaluateModels()
