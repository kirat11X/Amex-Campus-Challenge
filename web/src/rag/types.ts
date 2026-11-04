export interface Chunk {
  chunk_id: string;
  source: string;
  kind: string;
  location: string;
  text: string;
}

export interface Hit extends Chunk {
  score: number;
}

/** What the search_knowledge_base tool needs from the retrieval layer. */
export interface KnowledgeBase {
  userChunkCount(): number;
  /** Top-k chunks above the similarity floor, best first. Rejects if embedding is unavailable. */
  search(query: string, corpus: 'core' | 'user', k: number): Promise<Hit[]>;
}
