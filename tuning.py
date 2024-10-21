import json
import os
import time
import typing

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

for model_info in genai.list_tuned_models():
    print(model_info.name)


class Reviews(typing.TypedDict):
    timestamp: int
    score: int


async def async_analyse_reviews(data: str):
    with open("./prompt.md", "r") as file:
        prompt = file.read() + "\n" + data

        model = genai.GenerativeModel(
            "gemini-1.5-pro",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[Reviews],
                max_output_tokens=8192,
            ),
        )

    for _ in range(3):
        try:
            response = await model.generate_content_async(prompt)
            response = json.loads(response.text)
            return response
        except json.JSONDecodeError:
            # logger.info("Failed to parse response. Skipping...")
            break
        except exceptions.ResourceExhausted:
            n_seconds = 3
            # logger.info(f"Rate limited. Waiting {n_seconds}s and retrying...")
            time.sleep(n_seconds)
            continue
