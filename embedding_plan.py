import csv
import nltk
import asyncio
openai_api_key = "sk-"

async def calculate_embeddings(sentences):
    client = AsyncOpenAI(api_key=openai_api_key)
    """Calculates embeddings for 1, 2, and 3-sentence chunks."""
    num_sentences = len(sentences)
    groupings = []
    groupings.extend(sentences)  # 1-sentence
    groupings.extend([" ".join(sentences[i:i+2]) for i in range(num_sentences - 1)]) # 2-sentence
    groupings.extend([" ".join(sentences[i:i+3]) for i in range(num_sentences - 2)]) # 3-sentence

    res = await client.embeddings.create(
        input=groupings,
        model="text-embedding-ada-002",
    )
    embeddings = [item.embedding for item in res.data]

    return {
        "1": embeddings[:num_sentences],
        "2": embeddings[num_sentences:num_sentences + num_sentences - 1],
        "3": embeddings[num_sentences + num_sentences - 1:]
    }

with open("1.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

sentences = nltk.sent_tokenize(raw_text)

embeddings = asyncio.run(calculate_embeddings(sentences))
print(len(embeddings["1"]))
print(len(embeddings["2"]))
print(len(embeddings["3"]))
