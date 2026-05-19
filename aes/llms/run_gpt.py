import argparse
import json

import openai

API_KEY = "" # OpenAI API Key
openai.api_key = API_KEY

SYSTEM_PROMPT_EN = """
You are an {language} essay scoring system.

Your task is score each essay according to the CEFR (Common European Framework of Reference) writing scale.
The CEFR scale uses six levels: A1, A2, B1, B2, C1, C2. Here are the guidelines for each level:
- A1 (Beginner): Can write simple isolated phrases and sentences.
- A2 (Elementary): Can write a series of simple phrases and sentences linked with simple connectors like 'and', 'but' and 'because'.
- B1 (Intermediate): Can write straightforward connected texts on a range of familiar subjects by linking a series of phrases.
- B2 (Upper-Intermediate): Can write clear, detailed texts on a variety of subjects, synthesising and evaluating information and arguments from a number of sources.
- C1 (Advanced): Can write clear, well-structured, and detailed text on complex subjects, expressing their views in a way that shows a mastery of text-forming strategies.
- C2 (Proficient): Can write clear, well-structured, and detailed text on complex subjects, mastering a variety of text-forming strategies. At this expert level, a user can write with fluency and accuracy. 

You will receive examples and inputs in JSON format and must always return 
a JSON object with the schema:
{{
  "input": "<input sentence>",
  "output": "<CEFR label>"
}}
If either the input or output is missing, your answer is invalid.

Guidelines:
1. Make sure to follow the CEFR guidelines above.
2. Output **only** valid JSON, no extra text, comments, or explanations.
3. The output label must be one of the six CEFR levels. 
"""

USER_PROMPT_EN = """
Return a JSON object with both 'input' and 'output' fields.
The 'input' field must contain the input sentence,
and the 'output' field must contain the CEFR label.
Here is the sentence:
"""

SYSTEM_PROMPT_AR = """
أنت نظام لتقييم المقالات المكتوبة باللغة {language}.

مهمتك هي تقييم كل مقالة وفقا لمقياس الكتابة الخاص بإطار CEFR
(Common European Framework of Reference).
يستخدم مقياس CEFR ستة مستويات: A1، A2، B1، B2، C1، C2.
فيما يلي الإرشادات الخاصة بكل مستوى:

- A1 (Beginner): يستطيع كتابة عبارات وجمل بسيطة ومعزولة.
- A2 (Elementary): يستطيع كتابة سلسلة من العبارات والجمل البسيطة المترابطة بروابط بسيطة مثل "و" و "لكن" و "بسبب".
- B1 (Intermediate): يستطيع كتابة نصوص مترابطة وبسيطة حول مواضيع مألوفة، من خلال ربط سلسلة من العبارات معا.
- B2 (Upper-Intermediate): يستطيع كتابة نصوص واضحة ومفصلة حول مجموعة متنوعة من المواضيع، مع القدرة على تلخيص وتقييم المعلومات والحجج من مصادر متعددة.
- C1 (Advanced): يستطيع كتابة نصوص واضحة ومنظمة ومفصلة حول مواضيع معقدة، والتعبير عن آرائه بطريقة تظهر إتقانه لاستراتيجيات بناء النصوص.
- C2 (Proficient): يستطيع كتابة نصوص واضحة ومنظمة ومفصلة حول مواضيع معقدة، مع إتقان مجموعة متنوعة من استراتيجيات بناء النصوص. في هذا المستوى المتقدم جدا، يمكن للكاتب الكتابة بطلاقة ودقة.

ستتلقى أمثلة ومدخلات بصيغة JSON، ويجب عليك دائمًا إخراج كائن JSON بالبنية التالية:
{{
  "input": "<نص الإدخال>",
  "output": "<تسمية مستوى CEFR>"
}}

إذا كان أي من الحقلين "input" أو "output" مفقودا، فإن إجابتك غير صالحة.

التعليمات:
1. تأكد من اتباع إرشادات CEFR الموضحة أعلاه.
2. أخرج النتيجة بصيغة JSON فقط، دون أي نص إضافي أو شرح.
3. يجب أن تكون التسمية الناتجة واحدة من مستويات CEFR الستة.
"""

USER_PROMPT_AR = """
أرجع كائن JSON يحتوي على الحقلين 'input' و 'output'.
يجب أن يحتوي الحقل 'input' على الجملة الأصلية،
بينما يجب أن يحتوي الحقل 'output' على تسمية ال CEFR.
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
                {"role": "user", "content": json.dumps(few_shot_examples, ensure_ascii=False)},
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
    
    write_data([x.strip() for x in outputs], args.output)
