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
### Instruction:أنت نظام لتقييم المقالات المكتوبة باللغة {language}.

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
### Input: {sent} \n### CEFR Label: "
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
        repetition_penalty=1.2,
        max_new_tokens=10,
        do_sample=True,
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True
    )[0]
    response = response.split("### CEFR Label:")[-1]
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