import openai
from transformers import pipeline
import llama
import os
import base64
# from eval_prompt import RESPONSE_PROMPT_4_TURBO, RESPONSE_PROMPT_GLOBAL_LOCAL_4_TURBO, RESPONSE_PROMPT_WITHOUT_RETRIEVE

def get_evaluator(model_path, eval_prompt):
    pipe = pipeline(model=model_path, device_map="auto")
    def evaluate(**kwargs):
        evaluation = pipe(
            eval_prompt.format(**kwargs),
            max_length=512,
            #do_sample=True,
            #temperature=0.2
        )
        return evaluation[0]["generated_text"]
    return evaluate

def get_gpt_evaluator(model_name, eval_prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    def evaluate(**kwargs):
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {"role": "user", "content": eval_prompt.format(**kwargs)}
            ]
        )
        return completion["choices"][0]["message"]["content"]
    return evaluate

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_gpt4v_response(model_name, prompt, vision, tactile, retrieved_caption=None, retrieved_caption_local=None):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    vision = encode_image(vision)
    tactile = encode_image(tactile)
    model = model_name
    if retrieved_caption is not None:
        if retrieved_caption_local is not None:
            input_prompt = RESPONSE_PROMPT_GLOBAL_LOCAL_4_TURBO.format(prompt=prompt, retrieved_caption=retrieved_caption, retrieved_caption_local=retrieved_caption_local)
            # input_prompt = RESPONSE_PROMPT_GLOBAL_LOCAL_4O_MINI.format(prompt=prompt, retrieved_caption=retrieved_caption, retrieved_caption_local=retrieved_caption_local)
        else:    
            input_prompt = RESPONSE_PROMPT_4_TURBO.format(prompt=prompt, retrieved_caption=retrieved_caption)
            # input_prompt = RESPONSE_PROMPT_4O_MINI.format(prompt=prompt, retrieved_caption=retrieved_caption)

        completion = openai.ChatCompletion.create(
            model=model,
            seed=42,
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for answering the quality of the answer. Limit your answers to a maximum of five tactile adjectives."},
                {"role": "user", "content": [
                    {"type": "text", "text": input_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{vision}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{tactile}"}},
                ]}
            ]
        )
        return completion["choices"][0]["message"]["content"]
    else:
        input_prompt = RESPONSE_PROMPT_WITHOUT_RETRIEVE.format(prompt=prompt)
        completion = openai.ChatCompletion.create(
            model=model,
            seed=42,
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for answering the quality of the answer."},
                {"role": "user", "content": [
                    {"type": "text", "text": input_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{vision}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{tactile}"}},
                ]}
            ]
        )
        return completion["choices"][0]["message"]["content"]



EVAL_MODEL = "lmsys/vicuna-33b-v1.3"

EVAL_PROMPT = """[User Question]: {prompt}\n\n
[Assistant Response]: {assistant_response}\n
[Correct Response]: {correct_response}\n\n
We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. 
The user asks the question on observing an image. The assistant's response is followed by the correct response.
\nPlease evaluate the assistant's response based on how closely it matches the correct response which describes tactile feelings. Please compare only the semantics of the answers. DO NOT consider grammatical errors in scoring the assistant. The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only one value indicating the score for the assistant. \nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.\n\n
"""

def load_model(model_path, llama_dir, args):
    if hasattr(args, "llama_type"):
        llama_type = args.llama_type
        print("llama_type", llama_type)
        return llama.load(model_path, llama_dir, llama_type=llama_type, args=args)
    return llama.load(model_path, llama_dir, args=args)

def load_single_db_model(model_path, llama_dir, args):
    if hasattr(args, "llama_type"):
        llama_type = args.llama_type
        return llama.llama_adapter_single_db.load(model_path, llama_dir, llama_type=llama_type, args=args)
    return llama.llama_adapter_single_db.load(model_path, llama_dir, args=args)

def load_global_local_model(model_path, llama_dir, args):
    if hasattr(args, "llama_type"):
        llama_type = args.llama_type
        return llama.llama_adapter_global_local.load(model_path, llama_dir, llama_type=llama_type, args=args)
    return llama.llama_adapter_global_local.load(model_path, llama_dir, args=args)

RESPONSE_PROMPT_4_TURBO = """
"Below is an instruction that describes a task."
"Write a response that appropriately completes the request."
"Limit your answers to a maximum of five tactile adjectives."
"Focus on using words commonly associated with the object from the reference list. Do not include words that do not relate to the object, even if they are present in the reference."
"Ensure that only relevant words are selected, avoiding any unrelated terms, and keep the response as concise as possible. Prioritize words frequently seen in association with the object.\n\n"

"Include captions from two datasets:"
"- Global dataset: Focus on semantics and overall appearance of objects."
"- Local dataset: Focus on texture-related expressions.\n\n"

"# Output Format\n\n"
"Your response should be a brief phrase or list that uses only words relevant to the object, excluding any unrelated terms from the reference.\n\n"

"### Instruction:\n{prompt}\n\n"
"### Reference words:\n{retrieved_caption}\n\n"
"### Response:"
"""

RESPONSE_PROMPT_4O_MINI = """
"Below is an instruction that describes a task."
"Write a response that appropriately completes the request."
"Given RGB image and tactile image, create a response by integrating and understanding both to represent the object comprehensively."
"Limit your answers to a maximum of five tactile adjectives."
"Focus on using words commonly associated with the object from the reference list. Do not include words that do not relate to the object, even if they are present in the reference."
"Ensure that only relevant words are selected, avoiding any unrelated terms, and keep the response as concise as possible. Prioritize words frequently seen in association with the object.\n\n"

"Include captions from two datasets:"
"- Global dataset: Focus on semantics and overall appearance of objects."
"- Local dataset: Focus on texture-related expressions.\n\n"

"# Output Format\n\n"
"Your response should be a brief phrase or list that uses only words relevant to the object, excluding any unrelated terms from the reference.\n\n"

"# Notes\n\n"
"- Do not include "shiny" or imply shininess when describing tactile images, as it is not representative of tactile characteristics."
"- Ensure descriptors are directly associated with the tactile nature of the object.\n\n"

"### Instruction:\n{prompt}\n\n"
"### Reference words:\n{retrieved_caption}\n\n"
"### Response:"
"""

# RESPONSE_PROMPT= """
# "Given an RGB image and a tactile image, create a response by integrating and understanding both to represent the object comprehensively.\n\n"

# "Focus on using words commonly associated with the object from the reference list. Do not include words that do not relate to the object, even if they are present in the reference. Ensure that only relevant words are selected, avoiding any unrelated terms, and keep the response as concise as possible. Prioritize words frequently seen in association with the object.\n\n"

# "# Steps\n\n"

# "1. Analyze the RGB and tactile images to understand the object visually and tactically."
# "2. Use up to five tactile adjectives to describe the object, focusing on relevant tactile characteristics."
# "3. Integrate these insights to determine the most representative tactile words."
# "4. Check the reference words list and ensure that irrelevant words are not included in the description."
# "5. Formulate a concise response.\n\n"

# "# Output Format\n\n"

# "Your response should be a brief phrase or list that uses only words relevant to the object, excluding any unrelated terms from the reference.\n\n"

# "# Notes\n\n"
# "- Do not include "shiny" or imply shininess when describing tactile images, as it is not representative of tactile characteristics."
# "- Ensure descriptors are directly associated with the tactile nature of the object.\n\n"

# "### Instruction:\n"
# "{prompt}\n\n"

# "### Reference words:\n"
# "{retrieved_caption}\n\n"

# "### Response:\n\n"
# """

RESPONSE_PROMPT_GLOBAL_LOCAL_4O_MINI = """
"Below is an instruction that describes a task."
"Write a response that appropriately completes the request."
"Given RGB image and tactile image, create a response by integrating and understanding both to represent the object comprehensively."
"Limit your answers to a maximum of five tactile adjectives."
"Focus on using words commonly associated with the object from the reference list. Do not include words that do not relate to the object, even if they are present in the reference."
"Ensure that only relevant words are selected, avoiding any unrelated terms, and keep the response as concise as possible. Prioritize words frequently seen in association with the object.\n\n"

"Include captions from two datasets:"
"- Global dataset: Focus on semantics and overall appearance of objects."
"- Local dataset: Focus on texture-related expressions.\n\n"

"# Output Format\n\n"
"Your response should be a brief phrase or list that uses only words relevant to the object, excluding any unrelated terms from the reference.\n\n"

"# Notes\n\n"
"- Do not include "shiny" or imply shininess when describing tactile images, as it is not representative of tactile characteristics."
"- Ensure descriptors are directly associated with the tactile nature of the object.\n\n"

"### Instruction:\n{prompt}\n\n"
"### Reference Global Datasets words:\n{retrieved_caption}\n\n"
"### Reference Local Datasets words:\n{retrieved_caption_local}\n\n"
"### Response:"
"""

RESPONSE_PROMPT_GLOBAL_LOCAL_4_TURBO = """
"Below is an instruction that describes a task."
"Write a response that appropriately completes the request."
"Limit your answers to a maximum of five tactile adjectives."
"Focus on using words commonly associated with the object from the reference list. Do not include words that do not relate to the object, even if they are present in the reference."
"Ensure that only relevant words are selected, avoiding any unrelated terms, and keep the response as concise as possible. Prioritize words frequently seen in association with the object.\n\n"

"Include captions from two datasets:"
"- Global dataset: Focus on semantics and overall appearance of objects."
"- Local dataset: Focus on texture-related expressions.\n\n"

"# Output Format\n\n"
"Your response should be a brief phrase or list that uses only words relevant to the object, excluding any unrelated terms from the reference.\n\n"

"### Instruction:\n{prompt}\n\n"
"### Reference Global Datasets words:\n{retrieved_caption}\n\n"
"### Reference Local Datasets words:\n{retrieved_caption_local}\n\n"
"### Response:"
"""

RESPONSE_PROMPT_WITHOUT_RETRIEVE_4O_MINI = """
"Below is an instruction that describes a task."
"Write a response that appropriately completes the request."
"Given RGB image and tactile image, create a response by integrating and understanding both to represent the object comprehensively."
"Limit your answers to a maximum of five tactile adjectives."
"Ensure that the response is concise and relevant to the object."
"Focus on using words commonly associated with the object from the reference list. Do not include words that do not relate to the object, even if they are present in the reference."
"Ensure that only relevant words are selected, avoiding any unrelated terms, and keep the response as concise as possible. Prioritize words frequently seen in association with the object.\n\n"

"# Output Format\n\n"
"Your response should be a brief phrase or list that uses only words relevant to the object, excluding any unrelated terms from the reference.\n\n"

"# Notes\n\n"
"- Do not include "shiny" or imply shininess when describing tactile images, as it is not representative of tactile characteristics."
"- Ensure descriptors are directly associated with the tactile nature of the object.\n\n"

"### Instruction:\n{prompt}\n\n"
"### Response:"
"""

RESPONSE_PROMPT_WITHOUT_RETRIEVE = """
"Below is an instruction that describes a task."
"Write a response that appropriately completes the request."
"Limit your answers to a maximum of five tactile adjectives."
"Focus on using words commonly associated with the object from the reference list. Do not include words that do not relate to the object, even if they are present in the reference."
"Ensure that only relevant words are selected, avoiding any unrelated terms, and keep the response as concise as possible. Prioritize words frequently seen in association with the object.\n\n"

"# Output Format\n\n"
"Your response should be a brief phrase or list that uses only words relevant to the object, excluding any unrelated terms from the reference.\n\n"

"### Instruction:\n{prompt}\n\n"
"### Response:"
"""