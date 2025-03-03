import csv
import nltk
import asyncio
openai_api_key = "sk-"

async def calculate_embeddings(sentences):
    client = AsyncOpenAI(api_key=openai_api_key)
    """Calculates embeddings for 1, 2, and 3-sentence chunks."""
    all_texts = []
    # 1-sentence embeddings
    all_texts.extend(sentences)
    # 2-sentence embeddings
    for i in range(len(sentences) - 1):
        all_texts.append(" ".join(sentences[i:i+2]))
    # 3-sentence embeddings
    for i in range(len(sentences) - 2):
        all_texts.append(" ".join(sentences[i:i+3]))

    res = await client.embeddings.create(
        input=all_texts,
        model="text-embedding-ada-002",
    )
    embeddings = [item.embedding for item in res.data]

    return {
        "1": embeddings[:len(sentences)],
        "2": embeddings[len(sentences):len(sentences) + len(sentences) - 1],
        "3": embeddings[len(sentences) + len(sentences) - 1:]
    }

with open("1.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

sentences = nltk.sent_tokenize(raw_text)

embeddings = asyncio.run(calculate_embeddings(sentences))
print(len(embeddings["1"]))
print(len(embeddings["2"]))
print(len(embeddings["3"]))
