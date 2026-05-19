import argparse
import json

import openai



SYSTEM_PROMPT_AR = """
أنت أداة لتصحيح الأخطاء النحوية والإملائية في اللغة {language}.

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
إليك الجملة:
"""

LANG_MAP = {
    ('en', 'en'): 'English',
    ('en', 'ar'): 'Arabic',
    ('ar', 'en'): 'الإنجليزية',
    ('ar', 'ar'): 'العربية'
}

client = openai.OpenAI(api_key='',
                       base_url='https://api.fanar.qa/v1')

def prompt(model, input_example, few_shot_examples, prompt_lang='en', data_lang='ar'):
    lang = LANG_MAP[(prompt_lang, data_lang)]

    # Selecting and format system/user prompts based on the prompt and data langs
    SYSTEM_PROMPT = SYSTEM_PROMPT_AR.format(language=lang)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(few_shot_examples, ensure_ascii=False),},
                {"role": "user", "content": (f"{USER_PROMPT_AR} {input_example['sent']}")}
            ],
            temperature=0,
            top_p=1
        )
        result = response.choices[0].message.content
        print(f'Example: {input_example["id"]} ... done!', flush=True)
        return result
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
            f.write(example.strip())
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
    

    write_data([x.strip() for x in outputs], args.output)
