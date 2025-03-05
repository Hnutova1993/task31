import numpy as np
import nltk
from openai import AsyncOpenAI
import time
import asyncio

class TextChunker:
    """
    A class to chunk text into semantically coherent segments using embeddings.

    Attributes:
        chunk_size (int): The maximum size (in characters) of a chunk.
        chunk_qty (int): The desired number of chunks.
        dim_emb (int): The dimensionality of the sentence embeddings (default: 1536).
        dim_qty (int):  A constant used for subchunk calculations (default: 20).
        client (AsyncOpenAI): Asynchronous OpenAI client for generating embeddings.

    Methods:
        get_dot(subchunk_qtys, subchunk_embeddings, ind1, ind2): Calculates the dot product between subchunks.
        get_dot2(chunk_qty, subchunk_qtys, subchunk_embeddings, ind1, ind2): Calculates intra- and inter-chunk dot products.
        get_subchunk(chunk_start, chunk_end, sent_embeddings): Creates subchunk embeddings within a chunk.
        get_reward1(chunks, chunk_qty, sent_embeddings): Calculates the initial reward for a chunking configuration.
        update_reward(chunks0, chunk_lengths0, chunk_size, chunk_qty, sent_lengths, sent_embeddings, ind, reward0, subchunk_qtys0, subchunk_embeddings0, dot0):  Updates the reward and chunking configuration.
        chunking(text): Performs the text chunking process.
    """

    def __init__(self, chunk_size, chunk_qty, api_key="YOUR_API_KEY"):
        """
        Initializes the TextChunker.

        Args:
            chunk_size (int): The maximum size of a chunk (in characters).
            chunk_qty (int): The desired number of chunks.
            api_key (str): Your OpenAI API key.  Defaults to a placeholder; MUST be replaced.
        """
        self.chunk_size = chunk_size
        self.chunk_qty = chunk_qty
        self.dim_emb = 1536
        self.dim_qty = 20
        self.client = AsyncOpenAI(api_key=api_key)
        self.INTRA = 0
        self.INTRA_NO = 1
        self.INTER = 2
        self.INTER_NO = 3


    def get_dot(self, subchunk_qtys, subchunk_embeddings, ind1, ind2):
        """Calculates a dot product related metric between subchunks."""
        dot = [0, 0]
        es1 = ind1 * self.dim_qty
        es2 = ind2 * self.dim_qty
        for j1 in range(subchunk_qtys[ind1]):
            for j2 in range(subchunk_qtys[ind2]):
                if (ind1 == ind2) and (j1 == j2):
                    continue
                dot[0] += np.dot(subchunk_embeddings[es1 + j1], subchunk_embeddings[es2 + j2])
                dot[1] += 1
        return dot

    def get_dot2(self, chunk_qty, subchunk_qtys, subchunk_embeddings, ind1, ind2):
        """Calculates intra- and inter-chunk dot products and their counts."""
        dot = [0, 0, 0, 0]
        dot1 = self.get_dot(subchunk_qtys, subchunk_embeddings, ind1, ind1)
        dot2 = self.get_dot(subchunk_qtys, subchunk_embeddings, ind2, ind2)
        dot[self.INTRA] = dot1[0] + dot2[0]
        dot[self.INTRA_NO] = dot1[1] + dot2[1]

        for j in range(chunk_qty):
            if j != ind1:
                dot1 = self.get_dot(subchunk_qtys, subchunk_embeddings, ind1, j)
                dot[self.INTER] += dot1[0]
                dot[self.INTER_NO] += dot1[1]
            if j != ind2:
                dot1 = self.get_dot(subchunk_qtys, subchunk_embeddings, ind2, j)
                dot[self.INTER] += dot1[0]
                dot[self.INTER_NO] += dot1[1]
        return dot

    def get_subchunk(self, chunk_start, chunk_end, sent_embeddings):
        """Creates subchunk embeddings by averaging sentence embeddings."""
        dim_qty1 = ((chunk_end - chunk_start) // 3) + 1
        subchunk_embeddings = np.zeros((dim_qty1, self.dim_emb))  # Use numpy array
        subchunk_qty = 0

        for j in range(chunk_start, chunk_end - 1, 3):
            subchunk_embeddings[subchunk_qty] = np.mean(sent_embeddings[j:j+3], axis=0)
            subchunk_qty += 1

        k = (chunk_end - chunk_start + 1) % 3
        if k == 1:
            subchunk_embeddings[subchunk_qty] = sent_embeddings[chunk_end]
        elif k == 2:
            subchunk_embeddings[subchunk_qty] = np.mean(sent_embeddings[chunk_end-1:chunk_end+1], axis=0)
        return subchunk_embeddings

    def get_reward1(self, chunks, chunk_qty, sent_embeddings):
        """Calculates the initial reward based on intra- and inter-chunk similarity."""
        subchunk_qtys = [0] * chunk_qty
        subchunk_embeddings = np.zeros((chunk_qty * self.dim_qty, self.dim_emb)) # Use numpy array
        dot = [0, 0, 0, 0]

        js = 0
        es = 0
        for i in range(chunk_qty):
            subchunk_qtys[i] = ((chunks[i] - js) // 3) + 1
            subchunk_embeddings[es : es + subchunk_qtys[i]] = self.get_subchunk(js, chunks[i], sent_embeddings)
            js = chunks[i] + 1
            es += self.dim_qty

            dot1 = self.get_dot(subchunk_qtys, subchunk_embeddings, i, i)
            dot[self.INTRA] += dot1[0]
            dot[self.INTRA_NO] += dot1[1]
            for i1 in range(i):
                dot1 = self.get_dot(subchunk_qtys, subchunk_embeddings, i, i1)
                dot[self.INTER] += dot1[0]
                dot[self.INTER_NO] += dot1[1]

        m_inter_dot = dot[self.INTER] / dot[self.INTER_NO] if dot[self.INTER_NO] != 0 else 0
        m_intra_dot = dot[self.INTRA] / dot[self.INTRA_NO] if dot[self.INTRA_NO] != 0 else 0
        reward = m_intra_dot - m_inter_dot

        result = {
            0: reward,
            1: subchunk_qtys,
            2: subchunk_embeddings,
            3: dot,
        }
        return result

    def update_reward(
        self,
        chunks0,
        chunk_lengths0,
        chunk_size,
        chunk_qty,
        sent_lengths,
        sent_embeddings,
        ind,
        reward0,
        subchunk_qtys0,
        subchunk_embeddings0,
        dot0,
    ):
        """Iteratively updates the chunking to maximize the reward."""
        js = chunks0[ind - 1] + 1 if ind > 0 else 0
        es = ind * self.dim_qty
        jk = 1

        chunk_ind = chunks0[ind]
        chunk_lengths2 = [chunk_lengths0[ind], chunk_lengths0[ind + 1]]
        reward = reward0
        subchunk_qtys2 = [subchunk_qtys0[ind], subchunk_qtys0[ind + 1]]

        # Use .copy() to create independent copies for numpy arrays
        subchunk_embeddings2 = subchunk_embeddings0[es : es + 2 * self.dim_qty].copy()

        dot = dot0.copy()

        dot1 = self.get_dot2(chunk_qty, subchunk_qtys0, subchunk_embeddings0, ind, ind + 1)
        for j in range(4):
            dot[j] -= dot1[j]

        chunk_length_ind = chunk_lengths0[ind]
        chunk_length1 = chunk_lengths0[ind + 1]
        for i in range(chunks0[ind], js + jk - 1, -1):
            if chunk_length1 + sent_lengths[i] > chunk_size:
                break
            chunk_length1 += sent_lengths[i]
            subchunk_qtys0[ind] = ((i - 1 - js) // 3) + 1
            subchunk_qtys0[ind + 1] = ((chunks0[ind + 1] - i) // 3) + 1
            subchunk_embeddings0[es : es + subchunk_qtys0[ind]] = self.get_subchunk(js, i - 1, sent_embeddings)
            subchunk_embeddings0[es + self.dim_qty : es + self.dim_qty + subchunk_qtys0[ind + 1]] = self.get_subchunk(i, chunks0[ind + 1], sent_embeddings)
            dot2 = self.get_dot2(chunk_qty, subchunk_qtys0, subchunk_embeddings0, ind, ind + 1)
            for j in range(4):
                dot2[j] += dot[j]
            m_inter_dot = dot2[self.INTER] / dot2[self.INTER_NO] if dot2[self.INTER_NO] != 0 else 0
            m_intra_dot = dot2[self.INTRA] / dot2[self.INTRA_NO] if dot2[self.INTRA_NO] != 0 else 0
            reward2 = m_intra_dot - m_inter_dot
            if reward2 > reward:
                chunk_ind = i - 1
                chunk_lengths2[0] = (chunk_lengths0[ind] + chunk_lengths0[ind + 1] - chunk_length1)
                chunk_lengths2[1] = chunk_length1
                reward = reward2
                subchunk_qtys2 = [subchunk_qtys0[ind], subchunk_qtys0[ind + 1]]
                subchunk_embeddings2[0 : subchunk_qtys0[ind]] = subchunk_embeddings0[es : es + subchunk_qtys0[ind]].copy()
                subchunk_embeddings2[self.dim_qty : self.dim_qty + subchunk_qtys0[ind + 1]] = subchunk_embeddings0[es + self.dim_qty : es + self.dim_qty + subchunk_qtys0[ind + 1]].copy()
                dot = dot2

        chunk_length1 = chunk_length_ind
        for i in range(chunks0[ind] + 1, chunks0[ind + 1] - jk + 1):
            if chunk_length1 + sent_lengths[i] > chunk_size:
                break
            chunk_length1 += sent_lengths[i]
            subchunk_qtys0[ind] = ((i - js) // 3) + 1
            subchunk_qtys0[ind + 1] = ((chunks0[ind + 1] - i - 1) // 3) + 1
            subchunk_embeddings0[es : es + subchunk_qtys0[ind]] = self.get_subchunk(js, i, sent_embeddings)
            subchunk_embeddings0[es + self.dim_qty : es + self.dim_qty + subchunk_qtys0[ind + 1]] = self.get_subchunk(i + 1, chunks0[ind + 1], sent_embeddings)
            dot2 = self.get_dot2(chunk_qty, subchunk_qtys0, subchunk_embeddings0, ind, ind + 1)
            for j in range(4):
                dot2[j] += dot[j]
            m_inter_dot = dot2[self.INTER] / dot2[self.INTER_NO] if dot2[self.INTER_NO] != 0 else 0
            m_intra_dot = dot2[self.INTRA] / dot2[self.INTRA_NO] if dot2[self.INTRA_NO] != 0 else 0
            reward2 = m_intra_dot - m_inter_dot
            if reward2 > reward:
                chunk_ind = i
                chunk_lengths2[0] = chunk_length1
                chunk_lengths2[1] = chunk_lengths0[ind] + chunk_lengths0[ind + 1] - chunk_length1
                reward = reward2
                subchunk_qtys2 = [subchunk_qtys0[ind], subchunk_qtys0[ind + 1]]
                subchunk_embeddings2[0 : subchunk_qtys0[ind]] = subchunk_embeddings0[es : es + subchunk_qtys0[ind]].copy()
                subchunk_embeddings2[self.dim_qty : self.dim_qty + subchunk_qtys0[ind + 1]] = subchunk_embeddings0[es + self.dim_qty : es + self.dim_qty + subchunk_qtys0[ind + 1]].copy()
                dot = dot2

        result = {
            0: chunk_ind,
            1: chunk_lengths2,
            2: reward,
            3: subchunk_qtys2,
            4: subchunk_embeddings2,
            5: dot,
        }
        return result

    async def chunking(self, text):
        """Performs the complete text chunking process."""
        sentences = nltk.sent_tokenize(text)
        sent_qty = len(sentences)
        sent_lengths = [len(sent) + 1 for sent in sentences]

        res = await self.client.embeddings.create(
            input=sentences, model="text-embedding-ada-002"
        )
        # Convert the embeddings to a numpy array immediately
        sent_embeddings = np.array([item.embedding for item in res.data])


        chunk_starts = []
        chunk_ends = []
        chunk_lengths = []
        sent3_qty = 0
        sent3_length = 0
        sent_cnt = 0
        cnt3 = 0
        while sent_cnt < sent_qty:
            if (sent3_length + sent_lengths[sent_cnt] <= self.chunk_size) and (cnt3 < 3):
                sent3_length += sent_lengths[sent_cnt]
                if cnt3 == 0:
                    chunk_starts.append(sent_cnt)
                sent_cnt += 1
                cnt3 += 1
            else:
                chunk_lengths.append(sent3_length)
                chunk_ends.append(sent_cnt - 1)
                sent3_qty += 1
                sent3_length = 0
                cnt3 = 0
        if cnt3 == 0:
            sent3_qty -= 1  # Correct: Adjust if the last group didn't form
        else:
            chunk_lengths.append(sent3_length)
            chunk_ends.append(sent_cnt - 1)

        for i in range(sent3_qty + 1, self.chunk_qty, -1):
            chunk_embeddings = np.zeros((i, self.dim_emb)) # use numpy array
            for j in range(i):
                # Use numpy array operations for summing embeddings
                chunk_embeddings[j] = np.mean(sent_embeddings[chunk_starts[j]:chunk_ends[j]+1], axis=0)


            max_dot = -1
            max_j = 0
            for j in range(i - 1):
                if chunk_lengths[j] + chunk_lengths[j + 1] <= self.chunk_size:
                    jdot = np.dot(chunk_embeddings[j], chunk_embeddings[j + 1])
                    if max_dot < jdot:
                        max_dot = jdot
                        max_j = j

            chunk_ends[max_j] = chunk_ends[max_j + 1]
            chunk_lengths[max_j] += chunk_lengths[max_j + 1]
            for j in range(max_j + 1, i - 1):
                chunk_starts[j] = chunk_starts[j + 1]
                chunk_ends[j] = chunk_ends[j + 1]
                chunk_lengths[j] = chunk_lengths[j + 1]
            chunk_starts.pop()
            chunk_ends.pop()
            chunk_lengths.pop()

        chunks = chunk_ends[0 : self.chunk_qty]
        result = self.get_reward1(chunks, self.chunk_qty, sent_embeddings)
        reward = result[0]
        subchunk_qtys = result[1]
        subchunk_embeddings = result[2]
        dot = result[3]

        for _ in range(2):  # Number of iterations
            for ind in range(self.chunk_qty - 1):
                es = ind * self.dim_qty
                result = self.update_reward(
                    chunks,
                    chunk_lengths,
                    self.chunk_size,
                    self.chunk_qty,
                    sent_lengths,
                    sent_embeddings,
                    ind,
                    reward,
                    subchunk_qtys,
                    subchunk_embeddings,
                    dot,
                )
                chunks[ind] = result[0]
                chunk_lengths[ind : ind + 2] = result[1]
                reward = result[2]
                subchunk_qtys[ind : ind + 2] = result[3]
                subchunk_embeddings[es : es + subchunk_qtys[ind]] = result[4][0 : subchunk_qtys[ind]]
                subchunk_embeddings[es + self.dim_qty : es + self.dim_qty + subchunk_qtys[ind + 1]] = result[4][self.dim_qty : self.dim_qty + subchunk_qtys[ind + 1]]
                dot = result[5]

        js = 0
        chunk_texts = []
        for i in range(self.chunk_qty):
            sent1 = " " + " ".join(sentences[js : chunks[i] + 1])
            chunk_texts.append(sent1)
            js = chunks[i] + 1

        return chunk_texts

async def main():
    index = 1
    # files = [{"name":"1.txt","chunk_size":2000,"num_chunks":32},{"name":"2.txt","chunk_size":4000,"num_chunks":18},{"name":"3.txt","chunk_size":4000,"num_chunks":16},{"name":"4.txt","chunk_size":2000,"num_chunks":32},{"name":"5.txt","chunk_size":3000,"num_chunks":14},{"name":"6.txt","chunk_size":3000,"num_chunks":16}]
    files = [{"name":"1.txt","chunk_size":2000,"num_chunks":32}]
    api_key = "sk-"
    for i in range(len(files)):

        with open(files[i]['name'],"r", encoding="utf-8") as file:
            text = file.read()

        chunker = TextChunker(chunk_size=files[i]['chunk_size'], chunk_qty=files[i]['num_chunks'], api_key=api_key)
        chunks = await chunker.chunking(text)
        for j, chunk in enumerate(chunks):
            print(f"chunk - {j+1} - {chunk}")
            print(len(chunk))
      
if __name__ == '__main__':
    asyncio.run(main())
