import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
from io import BytesIO
import time

url = "https://www.socialfocus.co.kr/news/photo/202207/14020_22505_4854.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

model = AutoModelForCausalLM.from_pretrained(
    "qresearch/llama-3.1-8B-vision-378",
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("qresearch/llama-3.1-8B-vision-378", use_fast=True)

start_time = time.time()
print(
    model.answer_question(
        image, "You are the monitoring manager of a fixed CCTV system. The input is real data collected from CCTV cameras installed at a port site, where the cameras are always fixed. To prevent potential or imminent safety accidents, evaluate the danger level by considering the relationship between key objects and people visible in the images. Predict the danger_score on a scale from 0 to 1. If there are no signs of a safety accident = 0, if a safety accident has occurred = 1. Output format: danger_score as a value between 0 and 1, and the reason for the danger_score in 20 words or less."
, tokenizer, max_new_tokens=128, do_sample=True, temperature=0.3
    ),
)
end_time = time.time()

print(f"{end_time - start_time:.5f} sec")
