

Incase you have received approval from meta website but waiting for HF approval. Here's how you can use the model directly from website. This should suffice for experimentation.


```bash

git clone https://github.com/meta-llama/llama.git
cd llama

# Paste the link from approval email (Subject: Get started with Llama 2). 
# Select 7B-chat
bash download.sh

mv tokenizer.model llama-2-7b-chat/tokenizer.model
mv tokenizer_checklist.chk llama-2-7b-chat/tokenizer_checklist.chk

wget https://raw.githubusercontent.com/huggingface/transformers/refs/heads/main/src/transformers/models/llama/convert_llama_weights_to_hf.py

# activate conda environment.
# ensure your environment has transformers, pytorch, sentencepiece protobuf installed.

python convert_llama_weights_to_hf.py --input_dir ./llama-2-7b-chat --model_size 7B --output_dir ./llamadownload --llama_version 2

```

Then replace model_id_1  with the path to llamadownload folder

Note for usage: You might get a warning, `Setting pad_token_id to eos_token_id:2 for open-end generation.` while running inference. 
Its harmless but you can get rid by setting `pad_token_id=tokenizer.eos_token_id`  in `model_obj.generate` call in `call_model` function.