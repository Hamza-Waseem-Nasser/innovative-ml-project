from transformers import pipeline

# Using a pretrained text-generation model
agent_pipeline = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

def get_agent_response(prompt: str) -> str:
    response = agent_pipeline(prompt, max_length=50)[0]['generated_text']
    return response

if __name__ == '__main__':
    prompt = "What if we increase the marketing budget by 10%?"
    print(get_agent_response(prompt))
