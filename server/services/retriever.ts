import { db } from '../db/database';
import { RetrievalResult } from '../db/models';
import { embeddingService } from './embeddingService';
import { vectorStore, SearchOptions } from './vectorStore';

export class Retriever {
  async retrieveRelevantChunks(query: string, options: SearchOptions): Promise<RetrievalResult[]> {
    if (!query || !query.trim()) {
      return [];
    }

    let effectiveDocumentId = options.documentId;
    if (options.documentId) {
      const doc = db.getDocument(options.documentId);
      if (!doc || doc.status !== 'COMPLETED') {
        effectiveDocumentId = undefined;
      }
    }

    // 2. Query Embedding (Requirement #24)
    const queryVector = await embeddingService.embedQuery(query.trim());

    // 3. Hybrid similarity search with user & document scoping (Requirement #25, #38, #39)
    const results = vectorStore.similaritySearch(queryVector, {
      ...options,
      documentId: effectiveDocumentId,
      queryText: query.trim(),
    });

    return results;
  }
}

export const retriever = new Retriever();
