import argparse
import json
import random
import time

import openai
from openai import RateLimitError, APIError

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

ستتلقى أمثلة ومدخلات بصيغة JSON:
{{
  "input": "<نص الإدخال>",
  "output": "<تسمية مستوى CEFR>"
}}

التعليمات:
1. تأكد من اتباع إرشادات CEFR الموضحة أعلاه.
2. أخرج النتيجة فقط، دون أي نص إضافي أو شرح.
3. يجب أن تكون التسمية الناتجة واحدة من مستويات CEFR الستة.
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

client = openai.OpenAI(api_key='PJ0fI4XXEurvSNFGKJ348G0F0EzFwAwd',
                       base_url='https://api.fanar.qa/v1')




def prompt(model, input_example, few_shot_examples, prompt_lang='en', data_lang='ar', max_retries=5):
    lang = LANG_MAP[(prompt_lang, data_lang)]
    SYSTEM_PROMPT = SYSTEM_PROMPT_AR.format(language=lang)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(few_shot_examples, ensure_ascii=False)},
        {"role": "user", "content": f"{USER_PROMPT_AR} {input_example['sent']}"}
    ]

    # Exponential backoff with random jitter
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                top_p=1,
            )
            result = response.choices[0].message.content
            print(f"Example {input_example['id']} ... done!", flush=True)
            return result

        except RateLimitError:
            sleep_time = 2 ** attempt + random.uniform(0, 1)
            print(f"Rate limit hit. Sleeping {sleep_time:.1f}s before retry ({attempt+1}/{max_retries})...")
            time.sleep(sleep_time)
            continue

        except APIError as e:
            sleep_time = 2 ** attempt + random.uniform(0, 1)
            print(f"Transient API error ({type(e).__name__}). Sleeping {sleep_time:.1f}s before retry...")
            time.sleep(sleep_time)
            continue

        except Exception as e:
            print(f"Unexpected error on example {input_example['id']}: {e}", flush=True)
            return input_example.get("sent", "")

    # If all retries fail
    print(f"Failed after {max_retries} retries for example {input_example['id']}", flush=True)
    return 'B1'



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