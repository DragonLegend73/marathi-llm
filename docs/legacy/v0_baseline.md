# v0 Baseline Documentation & Results

This document archives the original (v0) baseline metrics, configurations, and evaluation results as reported in the initial README prior to the rigorous benchmark upgrade plan.

---

## 1. Overview & Setup (v0)

- **Base model**: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`
- **Dataset**: `rachittshah/alpaca-marahti` (52,002 total samples, 34,499 after length filtering at `MAX_SEQ_LEN=512`)
- **Method**: QLoRA (r=16, alpha=32, dropout=0, auto target modules)
- **Epochs**: 1.0 (4,097 optimizer steps at effective batch size 8 = per-device batch 1 × grad_accum 8)
- **Optimizer**: `paged_adamw_8bit`, learning rate `2e-4`, warmup ratio `0.05`
- **Loss Masking**: None (trained on full text: prompt + completion together via `dataset_text_field="text"`)
- **Reported Final Train Loss**: 0.46 (started at 1.36)
- **Reported Final Validation Loss**: 0.4479
- **Public Adapter**: `DragonLegend/marathi-qwen2.5-lora`

---

## 2. Reported v0 Results

### Training Loss Progress
- Initial training loss: ~1.36
- Final training loss: ~0.46
- Final validation loss: 0.4479 (evaluated every 500 steps)

### Reported chrF++ Evaluation (10 Prompts)
Evaluation was run on 10 hand-selected Marathi prompts decoded with sampling (`temperature=0.7`, `max_new_tokens=256`), scored with `sacrebleu` CHRF (word_order=2) against 10 custom hardcoded reference strings.

| # | Prompt | Base chrF++ | Fine-Tuned chrF++ | Δ |
|---|---|---|---|---|
| 1 | निरोगी राहण्यासाठी तीन टिप्स द्या. | 17.92 | 28.25 | +10.33 |
| 2 | तीन मुख्य रंग कोणते आहेत? | 9.57 | 42.65 | +33.07 |
| 3 | पाण्याचे महत्त्व सांगा. | 11.89 | 22.34 | +10.45 |
| 4 | भारताची राजधानी कोणती आहे? | 21.84 | 46.90 | +25.06 |
| 5 | सूर्यमाला म्हणजे काय? | 12.18 | 16.46 | +4.28 |
| 6 | मित्राला पत्र कसे लिहावे? | 14.62 | 17.31 | +2.69 |
| 7 | झाडे लावणे का महत्त्वाचे आहे? | 11.83 | 20.43 | +8.60 |
| 8 | संगणक म्हणजे काय आणि त्याचा उपयोग काय? | 17.41 | 20.22 | +2.81 |
| 9 | वेळेचे व्यवस्थापन कसे करावे? | 21.77 | 25.49 | +3.72 |
| 10 | चांगल्या विद्यार्थ्याचे गुण कोणते असतात? | 14.84 | 18.24 | +3.40 |
| **Corpus** | **Average** | **15.39** | **25.83** | **+10.44** |

---

## 3. Original Evaluation References (from `evaluate.py`)

1. `1. संतुलित आहार घ्या आणि भरपूर फळे आणि भाज्या खा. 2. नियमित व्यायाम करा. 3. पुरेशी झोप घ्या.`
2. `लाल, निळा आणि पिवळा असे तीन प्राथमिक रंग आहेत.`
3. `पाणी जीवनासाठी अत्यंत आवश्यक आहे. ते शरीरातील पेशींना कार्यक्षम ठेवते आणि पचनक्रिया सुधारते.`
4. `भारताची राजधानी नवी दिल्ली आहे.`
5. `सूर्यमाला म्हणजे सूर्य आणि त्याच्याभोवती फिरणारे ग्रह, उपग्रह, लघुग्रह आणि धूमकेतू यांचा समूह.`
6. `प्रिय मित्र, मी तुला हे पत्र लिहित आहे कारण मला तुझ्याशी काही महत्त्वाच्या गोष्टी सांगायच्या आहेत.`
7. `झाडे आपल्याला ऑक्सिजन देतात, पर्यावरण स्वच्छ ठेवतात आणि पाऊस आणण्यास मदत करतात.`
8. `संगणक एक इलेक्ट्रॉनिक उपकरण आहे जे माहिती साठवते, प्रक्रिया करते आणि विविध कामांसाठी वापरले जाते.`
9. `वेळेचे व्यवस्थापन करण्यासाठी दैनंदिन वेळापत्रक बनवा, प्राधान्यक्रम ठरवा आणि विलंब टाळा.`
10. `चांगला विद्यार्थी नियमित अभ्यास करतो, शिक्षकांचा आदर करतो आणि जिज्ञासू वृत्ती ठेवतो.`
