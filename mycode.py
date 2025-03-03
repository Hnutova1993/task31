import nltk
import asyncio
import numpy as np
from openai import AsyncOpenAI
import os
import openai
import time
import pandas as pd

nltk.download("punkt", quiet=True)

class Chunk:
    def __init__(self, sentences: list[str]):
        self.sentences = sentences
        self.text = " ".join(sentences)
        self.text_length = len(self.text)

async def compute_sentence_embeddings(sentences: list[str], api_key: str):
    """Computes and caches embeddings for each individual sentence."""
    sentence_embedding_cache = {}
    if not sentences:
        return sentence_embedding_cache

    client = AsyncOpenAI(api_key=api_key)
    try:
        res = await client.embeddings.create(
            input=sentences,
            model="text-embedding-ada-002"
        )
        embeddings = np.array([item.embedding for item in res.data])

        for i, sentence in enumerate(sentences):
            sentence_embedding_cache[sentence] = embeddings[i]

    except Exception as e:
        print(f"Error during embedding computation: {e}")
        return {}  # Return empty cache on error

    return sentence_embedding_cache

async def compute_reward(chunks: list[Chunk], sentence_embedding_cache: dict):
    """Computes reward using sentence embeddings, highly optimized."""
    if not chunks:
        return 0.0

    all_sentences = []
    for chunk in chunks:
        all_sentences.extend(chunk.sentences)

    # Convert sentences to a list of embeddings directly.  Handle missing sentences.
    all_embeddings = [sentence_embedding_cache.get(s) for s in all_sentences]
    valid_embeddings = [emb for emb in all_embeddings if emb is not None]

    if not valid_embeddings:  # Handle the case where NO sentences have embeddings
        return 0.0

    all_embeddings = np.array(valid_embeddings)
    
    # Create mapping from sentence index to chunk index
    sentence_to_chunk_indices = []
    sentence_index = 0
    for chunk_index, chunk in enumerate(chunks):
        for _ in chunk.sentences:
            if sentence_index < len(all_embeddings): # Make sure the embedding exists
                sentence_to_chunk_indices.append(chunk_index)
            sentence_index +=1

    # Ensure lengths match
    min_len = min(len(all_embeddings), len(sentence_to_chunk_indices))
    all_embeddings = all_embeddings[:min_len]
    sentence_to_chunk_indices = np.array(sentence_to_chunk_indices[:min_len])

    group_embeddings = []
    group_chunk_indices = []
    start_index = 0
    
    for chunk_index, chunk in enumerate(chunks):
      sentences_in_chunk = [s for s in chunk.sentences if s in sentence_embedding_cache] # Filter out sentences not in cache
      num_valid_sentences = len(sentences_in_chunk)
      for i in range(0, num_valid_sentences, 3):
        group_end = min(start_index + i + 3, len(all_embeddings))
        group_embedding = np.mean(all_embeddings[start_index + i:group_end], axis=0)
        group_embeddings.append(group_embedding)
        group_chunk_indices.append(chunk_index)
      start_index += len(chunk.sentences)  # Use original length to maintain correct indexing

    group_embeddings = np.array(group_embeddings)
    group_chunk_indices = np.array(group_chunk_indices)

    if len(group_embeddings) < 2:
        return 0.0

    # Vectorized similarity calculation
    similarities = np.dot(group_embeddings, group_embeddings.T)

    # Create a mask for intra-chunk and inter-chunk comparisons
    intra_mask = group_chunk_indices[:, None] == group_chunk_indices[None, :]
    # Ensure the diagonal is not included in intra_mask
    np.fill_diagonal(intra_mask, False)
    inter_mask = ~intra_mask

    intra_mean = np.mean(similarities[intra_mask]) if np.any(intra_mask) else 0.0
    inter_mean = np.mean(similarities[inter_mask]) if np.any(inter_mask) else 0.0

    return intra_mean - inter_mean
    
async def iterative_merge(chunks: list[Chunk], sentence_embedding_cache: dict, max_chunk_size: int, max_chunk_quantity: int):
    """Iteratively merges chunks, considering merging 2 or 3 adjacent chunks."""
    current_reward = await compute_reward(chunks, sentence_embedding_cache)

    while len(chunks) > max_chunk_quantity:
        best_reward_improvement = -float('inf')
        best_merge_indices = None  # Store a list of indices to merge
        best_chunks = []
        best_reward = -float('inf')
        di = -1
        for i in range(len(chunks) - 1):  # Iterate up to the second-to-last chunk
            # Try merging two chunks
            if i + 1 < len(chunks):
                merged_sentences_2 = chunks[i].sentences + chunks[i + 1].sentences
                if len(" ".join(merged_sentences_2)) <= max_chunk_size:
                    new_chunks_2 = chunks[:i] + [Chunk(merged_sentences_2)] + chunks[i + 2:]
                    new_reward_2 = await compute_reward(new_chunks_2, sentence_embedding_cache)
                    reward_improvement_2 = new_reward_2 - current_reward

                    if reward_improvement_2 > best_reward_improvement:
                        best_reward_improvement = reward_improvement_2
                        best_merge_indices = [i, i + 1]  # Merge indices i and i+1
                        best_chunks = new_chunks_2
                        best_reward = new_reward_2
                        di = 0

            # Try merging three chunks
            # if i + 2 < len(chunks):
            #     merged_sentences_3 = chunks[i].sentences + chunks[i + 1].sentences + chunks[i + 2].sentences
            #     if len(" ".join(merged_sentences_3)) <= max_chunk_size:
            #         new_chunks_3 = chunks[:i] + [Chunk(merged_sentences_3)] + chunks[i + 3:]
            #         new_reward_3 = await compute_reward(new_chunks_3, sentence_embedding_cache)
            #         reward_improvement_3 = new_reward_3 - current_reward

            #         if reward_improvement_3 > best_reward_improvement:
            #             best_reward_improvement = reward_improvement_3
            #             best_merge_indices = [i, i + 1, i + 2]  # Merge indices i, i+1, and i+2
            #             best_chunks = new_chunks_3
            #             best_reward = new_reward_3
            #             di = 1


        if best_merge_indices:
            chunks = best_chunks
            current_reward = best_reward
            print(
                f"Merged chunks at indices {best_merge_indices}. New reward: {current_reward:.4f}. Chunks: {len(chunks)}. direction: {di}")
        else:
            print("No beneficial merge found. Stopping iterations.")
            break

    return chunks


async def optimized_chunking(text: str, max_chunk_size: int, max_chunk_quantity: int, api_key: str):
    """Main chunking function."""
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    sentence_embedding_cache = await compute_sentence_embeddings(sentences, api_key)
    if not sentence_embedding_cache:
        return []

    initial_chunks = [Chunk([sentence]) for sentence in sentences]
    final_chunks = await iterative_merge(initial_chunks, sentence_embedding_cache, max_chunk_size, max_chunk_quantity)

    return final_chunks

# --- Example Usage ---
if __name__ == '__main__':

    index = 1
    files = [{"name":"1.txt","chunk_size":2000,"num_chunks":32},{"name":"2.txt","chunk_size":4000,"num_chunks":18},{"name":"3.txt","chunk_size":4000,"num_chunks":16},{"name":"4.txt","chunk_size":2000,"num_chunks":32},{"name":"5.txt","chunk_size":3000,"num_chunks":14},{"name":"6.txt","chunk_size":3000,"num_chunks":16}]
    api_key = "sk_"
    for i in range(len(files)):

        start_time = time.time()
        with open(files[i]['name'],"r", encoding="utf-8") as file:
            text = file.read()
        
        chunks = asyncio.run(optimized_chunking(text, files[i]['chunk_size'], files[i]['num_chunks'], api_key))

        chunks_file = []
        for j, chunk in enumerate(chunks):
            print(f"chunk - {j+1} - {chunk.text}")
            print(len(chunk.text))
            chunks_file.append({"Chunk Index":j+1,"Chunk Content":chunk.text})


        df = pd.DataFrame(chunks_file)

        output_file = f"re-{index}.csv"
        df.to_csv(output_file, index=False)  

        print("CSV file saved successfully!")
        end_time = time.time()
        index += 1
        print(f"{files[i]['name']} : {output_file}")
        print(f"{files[i]['name']} took {end_time - start_time}s to complete chunking")
