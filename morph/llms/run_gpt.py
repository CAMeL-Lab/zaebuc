import argparse
import json
from collections import defaultdict

import openai

API_KEY = "" # OpenAI API Key
openai.api_key = API_KEY

SYSTEM_PROMPT_EN = """
You are an {language} morphological analyzer.

Your task is provide the following morphological features for each input word in the given text:
- the tokenized word
- the part-of-speech (POS) tag according to the universal dependencies (UD) framework
- the lemma of the word

You will receive examples and inputs in JSON format and must always return 
a JSON object with the schema:
{{
  "input": "<input sentence>",
  "output": "<annotated sentence>"
}}
Each word in the annotated sentence has the following format: <w>input_word<w>tokenized_word<w>pos_tag<w>lemma<w>. Your output must follow this exact format
If either the input or output is missing from the output JSON, your answer is invalid.

Guidelines:
1. Make sure to follow the guidelines above.
2. Output **only** valid JSON, no extra text, comments, or explanations.
3. Each word in the output annotated sentence must follow the format above.
4. The output annotated sentence **must** have the same number words as the input sentence.
"""

USER_PROMPT_EN = """
Return a JSON object with both 'input' and 'output' fields.
The 'input' field must contain the input sentence,
and the 'output' field must contain the annotated sentence where each word is annotated with the morphological features specified above.
Here is the sentence:
"""


SYSTEM_PROMPT_ZERO_EN = """
You are an {language} morphological analzyer.

Your task is provide the following morphological features for each input word in the given text:
- the tokenized word
- the part-of-speech (POS) tag according to the universal dependencies (UD) framework
- the lemma of the word

You must always return a JSON object with the schema:
{{
  "input": "<input sentence>",
  "output": "<annotated sentence>"
}}

Each word in the annotated sentence has the following format: <w>input_word<w>tokenized_word<w>pos_tag<w>lemma<w>. Your output must follow this exact format
If either the input or output is missing from the output JSON, your answer is invalid.

Guidelines:
1. Make sure to follow the guidelines above.
2. Output **only** valid JSON, no extra text, comments, or explanations.
3. Each word in the output annotated sentence must follow the format above.
4. The output annotated sentence **must** have the same number words as the input sentence.
"""

USER_PROMPT_ZERO_EN = """
Return a JSON object with both 'input' and 'output' fields.
The 'input' field must contain the input sentence,
and the 'output' field must contain the annotated sentence where each word is annotated with the morphological features specified above.
Here is the sentence:
"""

SYSTEM_PROMPT_AR = """
أنت محلل صرفي للغة {language}.

مهمتك هي استخراج الخصائص الصرفية التالية لكل كلمة في النص المعطى:
- الكلمة بعد التجزئة
- universal dependencies (UD) القسم النحوي للكلمة وفق إطار 
- المدخل المعجمي للكلمة

ستتلقى أمثلة ومدخلات بصيغة JSON، ويجب عليك دائما إخراج كائن JSON بالبنية التالية:
{{
  "input": "<الجملة الأصلية>",
  "output": "<الجملة الموسمة>"
}}

كل كلمة في الجملة الموسمة يجب ان تكون بالصيغة التالية:
<w>الكلمة المدخلة<w>الكلمة بعد التجزئة<w>القسم النحوي<w>المدخل المعجمي<w>
ويجب ان يتبع الإخراج هذا الشكل تماما.

اذا كان أي من الحقلين "input" أو "output" مفقودا من كائن JSON الناتج، فإن الإجابة غير صالحة.

التعليمات:
1. تأكد من اتباع التعليمات المذكورة أعلاه بدقة.
2. أخرج النتيجة بصيغة JSON فقط، دون أي نص إضافي أو تعليقات أو شروحات.
3. كل كلمة في الجملة الموسمة يجب ان تتبع الصيغة المحددة أعلاه.
4. يجب ان تحتوي الجملة الموسمة على نفس عدد الكلمات الموجود في الجملة الأصلية.
"""

USER_PROMPT_AR = """
أرجع كائن JSON يحتوي على الحقلين "input" و "output".
يجب ان يحتوي الحقل "input" على الجملة الأصلية،
ويجب ان يحتوي الحقل "output" على الجملة الموسمة التي تم فيها تمييز كل كلمة بالخصائص الصرفية المحددة أعلاه.
إليك الجملة:
"""


LANG_MAP = {
    ('en', 'en'): 'English',
    ('en', 'ar'): 'Arabic',
    ('en', 'csw'): 'Arabic-English'
}

client = openai.OpenAI()

def prompt(model, input_example, few_shot_examples, prompt_lang='en', data_lang='ar'):
    lang = LANG_MAP[(prompt_lang, data_lang)]

    # Selecting and format system/user prompts based on the prompt and data langs
    if prompt_lang == 'en':
        SYSTEM_PROMPT = SYSTEM_PROMPT_EN.format(language=lang) if few_shot_examples is not None else SYSTEM_PROMPT_ZERO_EN.format(language=lang)
        USER_PROMPT = USER_PROMPT_EN if few_shot_examples is not None else USER_PROMPT_ZERO_EN
    else:
        SYSTEM_PROMPT = SYSTEM_PROMPT_AR.format(language=lang)
        USER_PROMPT = USER_PROMPT_AR

    try:
        if few_shot_examples is None:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (f"{USER_PROMPT}" +
                                             f"{json.dumps({'input': input_example['sent']}, ensure_ascii=False)}")}
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(few_shot_examples, ensure_ascii=False)},
                {"role": "user", "content": (f"{USER_PROMPT}" +
                                             f"{json.dumps({'input': input_example['sent']}, ensure_ascii=False)}")}
            ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
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


def load_data(path):
    docs = []
    doc = defaultdict(lambda: defaultdict(list))
    
    with open(path) as f:
        for i, line in enumerate(f.readlines()[1:]):
            line = line.strip()
            if line:
                doc_id, word, tok, pos, lemma = line.split('\t')
                doc[doc_id]['word'].append(word)
                doc[doc_id]['tok'].append(tok)
                doc[doc_id]['pos'].append(pos)
                doc[doc_id]['lemma'].append(lemma)
            else:
                docs.append(doc)
                doc = defaultdict(lambda: defaultdict(list))

        if doc:
            docs.append(doc)
    
    data = []
    for i, doc in enumerate(docs):
        doc_id = list(doc.keys())[0]
        data.append({'id': i, 'sent': ' '.join(doc[doc_id]['word'])})

    return data


def load_data_txt(path):
    data = []
    with open(path) as f:
        for i, line in enumerate(f.readlines()):
            data.append({'id': i, 'sent': line.strip()})
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
    parser.add_argument("--examples", default=None, help="few shot examples path.")
    parser.add_argument("--input_data", help="Input data file path.")
    parser.add_argument("--output", type=str, help="Output directory.")

    args = parser.parse_args()
    
    if args.examples is None: # spoken
        input_data = load_data_txt(args.input_data)
        few_shot_examples = None
    else: # written
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
