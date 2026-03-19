import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
import json

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


# model_path = "inceptionai/jais-family-13b-chat"
model_path = "inceptionai/jais-family-30b-8k-chat"
device = ('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True,
                                             torch_dtype=torch.bfloat16)


SYSTEM_PROMPT_AR = """
### Instruction:أنت أداة لتصحيح الأخطاء النحوية والإملائية في اللغة {language}.

مهمتك هي تحديد وتصحيح **الأخطاء النحوية والإملائية فقط** في الجمل {language} 
مع الحفاظ على المعنى الأصلي للسياق دون أي تغيير في المقصود.

سيتم تزويدك ببعض الأمثلة والمدخلات على شكل JSON بنفس البنية التالية:
{{
  "input": "<الجملة الأصلية>",
  "output": "<الجملة المصححة>"
}}

التعليمات:
1. أجر أقل عدد ممكن من التعديلات لتصحيح الجملة.
2. لا تعد صياغة الأجزاء الصحيحة نحويا أو أسلوبيا.
3. تجنب تغيير المعنى من خلال إضافة أو حذف أي المعلومات.
4. أخرج النص المصحح فقط، دون أي نص إضافي أو شرح أو علامات تنسيق.
"""

USER_PROMPT_AR = """
أكمل المحادثة بين [|Human|] و[|AI|] :\n### Input:[|Human|] {sent} \n[|AI|]\n### Response: "
"""

LANG_MAP = {
    ('en', 'en'): 'English',
    ('en', 'ar'): 'Arabic',
    ('ar', 'en'): 'الإنجليزية',
    ('ar', 'ar'): 'العربية'
}

def prompt(input_example, few_shot_examples, prompt_lang='en', data_lang='ar'):
    lang = LANG_MAP[(prompt_lang, data_lang)]

    # Selecting and format system/user prompts based on the prompt and data langs
    prompt = (SYSTEM_PROMPT_AR.format(language=lang) + '\n' + 
              json.dumps(few_shot_examples, ensure_ascii=False) + '\n' + 
              USER_PROMPT_AR.format(sent=input_example["sent"]))
    # print(prompt, flush=True)
    pred = inference_example(prompt)
    print(f'Example: {input_example["id"]} ... done!', flush=True)
    return pred


def inference_example(text):
    input_ids = tokenizer(text, return_tensors="pt", max_length=5000).input_ids
    inputs = input_ids.to(device)
    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=0.3,
        max_length=5000,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True
    )[0]
    response = response.split("### Response :")[-1]
    return response

# def inference(dataloader):
#     preds = []
#     for batch in dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}

#         input_len = batch['input_ids'].shape[-1]
#         generate_ids = model.generate(
#             **batch,
#             top_p=0.9,
#             temperature=0.3,
#             repetition_penalty=1.2,
#             do_sample=True,
#         )
#         response = tokenizer.batch_decode(
#             generate_ids, skip_special_tokens=True
#         )

#         preds.extend(response)
    
#     preds = [response.split("### Response :")[-1] for response in preds]
#     return preds


def load_data(src_path):
    data = []
    with open(src_path) as f:
        for sent_id, src_sent in enumerate(f.readlines()):
            data.append({'id': sent_id, 'sent': src_sent.strip()})
    return data


def write_data(data, output_path):
    with open(output_path, mode="w") as f:
        for example in data:
            f.write(example.strip())
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_lang", default='en', help="language to prompt the model in.")
    parser.add_argument("--data_lang", default='en', help="language of the data examples.")
    parser.add_argument("--examples", help="few shot examples path.")
    parser.add_argument("--input_data", help="Input data file path.")
    parser.add_argument("--output", type=str, help="Output directory.")

    args = parser.parse_args()
    
    input_data = load_data(args.input_data)
    with open(args.examples) as f:
        few_shot_examples = json.load(f)

    outputs = []
    for example in input_data:
        output = prompt(input_example=example, few_shot_examples=few_shot_examples,
                        prompt_lang=args.prompt_lang,
                        data_lang=args.data_lang)
        outputs.append(output)
    
    write_data([x.strip() for x in outputs], args.output)