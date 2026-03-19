import argparse
import json

import openai

from utils import pnx_tokenize_dediac

API_KEY = "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" # OpenAI API Key
openai.api_key = API_KEY

SYSTEM_PROMPT_EN = """
You are an {language} grammatical error correction tool.

Your task is to identify and correct **only grammatical and spelling errors** 
in {language} sentences, while preserving their original meaning and phrasing.

You will receive examples and inputs in JSON format and must always return 
a JSON object with the schema:
{{
  "input": "<original sentence>",
  "output": "<corrected sentence>"
}}
If either the input or output is missing, your answer is invalid.

Guidelines:
1. Make the minimal edits necessary to correct grammar or spelling.
2. Do not rephrase correct parts of the sentence.
3. Avoid altering the meaning by adding or removing information.
4. Output **only** valid JSON, no extra text, comments, or explanations.
"""
USER_PROMPT_EN = """
Return a JSON object with both 'input' and 'output' fields.
The 'input' field must contain the original sentence,
and the 'output' field must contain the corrected sentence.
Here is the sentence:
"""

SYSTEM_PROMPT_AR = """
أنت أداة لتصحيح الأخطاء النحوية والإملائية في اللغة {language}.

مهمتك هي تحديد وتصحيح **الأخطاء النحوية والإملائية فقط** في الجمل {language} 
مع الحفاظ على المعنى الأصلي للسياق دون أي تغيير في المقصود.

سيتم تزويدك ببعض الأمثلة والمدخلات على شكل JSON، ويجب عليك إخراج النتائج في صيغة JSON بنفس البنية التالية:
{{
  "input": "<الجملة الأصلية>",
  "output": "<الجملة المصححة>"
}}
إذا كان أي من الحقلين input أو output مفقودا، فإجابتك غير صالحة.

التعليمات:
1. أجر أقل عدد ممكن من التعديلات لتصحيح الجملة.
2. لا تعد صياغة الأجزاء الصحيحة نحويا أو أسلوبيا.
3. تجنب تغيير المعنى من خلال إضافة أو حذف أي المعلومات.
4. أخرج النتيجة بصيغة JSON فقط، دون أي نص إضافي أو شرح أو علامات تنسيق.
"""
USER_PROMPT_AR = """
أرجع كائن JSON يحتوي على الحقلين 'input' و 'output'.
يجب أن يحتوي الحقل 'input' على الجملة الأصلية،
بينما يجب أن يحتوي الحقل 'output' على الجملة المصحَّحة.
إليك الجملة:
"""

LANG_MAP = {
    ('en', 'en'): 'English',
    ('en', 'ar'): 'Arabic',
    ('ar', 'en'): 'الإنجليزية',
    ('ar', 'ar'): 'العربية'
}

client = openai.OpenAI()

def prompt(model, input_example, few_shot_examples, prompt_lang='en', data_lang='ar'):
    lang = LANG_MAP[(prompt_lang, data_lang)]

    # Selecting and format system/user prompts based on the prompt and data langs
    if prompt_lang == 'en':
        SYSTEM_PROMPT = SYSTEM_PROMPT_EN.format(language=lang)
        USER_PROMPT = USER_PROMPT_EN
    else:
        SYSTEM_PROMPT = SYSTEM_PROMPT_AR.format(language=lang)
        USER_PROMPT = USER_PROMPT_AR

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(few_shot_examples, ensure_ascii=False),},
                {"role": "user", "content": (f"{USER_PROMPT}" +
                                             f"{json.dumps({'input': input_example['sent']}, ensure_ascii=False)}")}
            ],
            response_format={"type": "json_object"},
            temperature=0 if model != 'gpt-5' else 1,
            top_p=1,
        )
        result = json.loads(response.choices[0].message.content)
        print(f'Example: {input_example["id"]} ... done!', flush=True)
        return result["output"]
    except:
        result = input_example['sent']
        print(f'ERROR! Example: {input_example["id"]}', flush=True)
        return result


def load_data(src_path):
    data = []
    with open(src_path) as f:
        for sent_id, src_sent in enumerate(f.readlines()):
            data.append({'id': sent_id, 'sent': src_sent.strip()})
    return data


def write_data(data, output_path):
    with open(output_path, mode="w") as f:
        for example in data:
            f.write(example)
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gpt-4o', help="OpenAI model.")
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
        output = prompt(model=args.model,
                        input_example=example, few_shot_examples=few_shot_examples,
                        prompt_lang=args.prompt_lang,
                        data_lang=args.data_lang)
        outputs.append(output)
    
    if args.data_lang == 'ar':
        pnx_tok_outputs = pnx_tokenize_dediac(outputs, dediac=True)
        write_data(pnx_tok_outputs, args.output)
    else:
        write_data([x.strip() for x in outputs], args.output)