// Mock data mirroring real papers from DESIGN.md RAGAS evaluations

export interface Paper {
  id: string
  title: string
  shortTitle: string
  authors: string[]
  year: number
  venue: string
  chunkCount: number
  sections: string[]
  fileSize: string
  addedAt: string
  abstract: string
}

export interface RetrievedChunk {
  chunkId: string
  paperId: string
  paperTitle: string
  shortTitle: string
  section: string
  text: string
  score: number
  subQuery: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  confidence?: 'high' | 'medium' | 'low'
  subQueries?: string[]
  replanCount?: number
  retrievedChunks?: RetrievedChunk[]
  citations?: Citation[]
  processingMs?: number
  cacheHit?: boolean
}

export interface Citation {
  id: string
  shortTitle: string
  section: string
  chunkId: string
}

export interface Conversation {
  id: string
  title: string
  createdAt: string
  messages: Message[]
}

// ── Papers ──────────────────────────────────────────────────────────────────

export const MOCK_PAPERS: Paper[] = [
  {
    id: 'p1',
    title: 'BEST-Route: Adaptive LLM Routing with Token Budget and Performance Constraints',
    shortTitle: 'BEST-Route',
    authors: ['Chen, W.', 'Liu, Y.', 'Zhang, R.'],
    year: 2024,
    venue: 'EMNLP',
    chunkCount: 87,
    sections: ['Abstract', 'Introduction', 'Related Work', 'Method', 'Experiments', 'Results', 'Conclusion', 'References'],
    fileSize: '1.2 MB',
    addedAt: '2025-01-14',
    abstract: 'We propose BEST-Route, a training-free adaptive LLM routing framework that optimizes token budget and performance simultaneously. Our router leverages query complexity signals to dispatch to the most appropriate LLM with empirical Pareto-optimal trade-offs.',
  },
  {
    id: 'p2',
    title: 'UniRoute: Unified Query Routing for Large Language Models',
    shortTitle: 'UniRoute',
    authors: ['Park, J.', 'Kim, S.', 'Lee, H.'],
    year: 2024,
    venue: 'ACL',
    chunkCount: 74,
    sections: ['Abstract', 'Introduction', 'Background', 'Unified Framework', 'Experimental Setup', 'Results', 'Analysis', 'Conclusion'],
    fileSize: '980 KB',
    addedAt: '2025-01-14',
    abstract: 'UniRoute presents a unified framework for query routing that handles both quality and efficiency constraints in multi-model LLM deployments. We introduce a novel scoring function that unifies prior routing criteria under a single objective.',
  },
  {
    id: 'p3',
    title: 'xRouter: Mixture-of-Experts LLM Routing via Expert Specialization',
    shortTitle: 'xRouter',
    authors: ['Wang, M.', 'Zhao, L.'],
    year: 2024,
    venue: 'NeurIPS Workshop',
    chunkCount: 62,
    sections: ['Abstract', 'Introduction', 'Method', 'Expert Specialization', 'Evaluation', 'Discussion'],
    fileSize: '750 KB',
    addedAt: '2025-01-15',
    abstract: 'xRouter uses mixture-of-experts principles to route queries to specialized LLMs. We show that expert-specialized routers outperform general-purpose approaches on domain-specific benchmarks.',
  },
  {
    id: 'p4',
    title: 'From BM25 to Corrective RAG: A Survey of Retrieval-Augmented Generation',
    shortTitle: 'From BM25 to Corrective RAG',
    authors: ['Johnson, A.', 'Smith, B.', 'Davis, C.', 'Brown, E.'],
    year: 2024,
    venue: 'ICLR',
    chunkCount: 134,
    sections: ['Abstract', 'Introduction', 'BM25 Era', 'Dense Retrieval', 'Hybrid Methods', 'Corrective RAG', 'Evaluation', 'Open Problems', 'Conclusion'],
    fileSize: '2.1 MB',
    addedAt: '2025-01-15',
    abstract: 'We survey the evolution of retrieval-augmented generation from sparse BM25 methods to modern corrective RAG approaches. We systematize evaluation protocols and identify open research directions for robust knowledge-grounded generation.',
  },
  {
    id: 'p5',
    title: 'GraphRAG: Graph-Enhanced Retrieval for Multi-Hop Reasoning',
    shortTitle: 'GraphRAG',
    authors: ['Thompson, R.', 'Nguyen, T.'],
    year: 2024,
    venue: 'SIGIR',
    chunkCount: 91,
    sections: ['Abstract', 'Introduction', 'Graph Construction', 'Multi-Hop Retrieval', 'Experiments', 'Ablation Study', 'Conclusion'],
    fileSize: '1.4 MB',
    addedAt: '2025-01-16',
    abstract: 'GraphRAG augments dense retrieval with structured knowledge graphs to enable multi-hop reasoning. We demonstrate significant improvements on complex questions requiring entity relationships across multiple documents.',
  },
  {
    id: 'p6',
    title: 'Conversational Memory for Long-Context LLMs: Compression and Retrieval',
    shortTitle: 'Conversational Memory',
    authors: ['Garcia, M.', 'Wilson, K.'],
    year: 2024,
    venue: 'EMNLP',
    chunkCount: 68,
    sections: ['Abstract', 'Introduction', 'Memory Architecture', 'Compression Strategies', 'Retrieval-Based Memory', 'Experiments', 'Conclusion'],
    fileSize: '890 KB',
    addedAt: '2025-01-16',
    abstract: 'We study memory compression and retrieval strategies for maintaining coherent multi-turn conversations with LLMs. Our approach achieves 4x context reduction with minimal degradation in answer quality.',
  },
]

// ── Conversations ────────────────────────────────────────────────────────────

export const MOCK_CONVERSATIONS: Conversation[] = [
  {
    id: 'conv1',
    title: 'Routing Strategies Comparison',
    createdAt: '2025-01-16T10:22:00Z',
    messages: [
      {
        id: 'm1',
        role: 'user',
        content: 'What is the primary generation metric used in BEST-Route and how does it compare to UniRoute\'s approach?',
        timestamp: '2025-01-16T10:22:00Z',
      },
      {
        id: 'm2',
        role: 'assistant',
        content: `BEST-Route and UniRoute use distinct primary generation metrics reflecting their different design philosophies.

**BEST-Route** employs a composite metric called the *Budget-Efficient Score with Tolerance* (BEST), defined as a weighted combination of task accuracy and token budget adherence [BEST-Route, §Method]. The tolerance parameter ε allows configurable trade-offs between performance and cost, making it adaptable to deployment constraints [BEST-Route, §Experiments].

**UniRoute**, by contrast, uses a *Unified Routing Score* (URS) that normalizes quality and efficiency into a single scalar via learned weights [UniRoute, §Unified Framework]. The key distinction is that UniRoute's weights are trained end-to-end, whereas BEST-Route's scoring is training-free [UniRoute, §Results].

**Note:** [BEST-Route] emphasizes token budget as a first-class constraint, while [UniRoute] treats efficiency as one component of a learned objective — these represent fundamentally different stances on how routing trade-offs should be specified.

Both systems achieve competitive accuracy on standard benchmarks, but BEST-Route reports 2.3× lower latency on simple queries due to its lightweight routing signal [BEST-Route, §Results].`,
        timestamp: '2025-01-16T10:22:28Z',
        confidence: 'high',
        processingMs: 4820,
        cacheHit: false,
        subQueries: [
          'What is the primary generation metric in BEST-Route?',
          'What metric does UniRoute use for routing decisions?',
          'How do BEST-Route and UniRoute compare on evaluation benchmarks?',
        ],
        replanCount: 0,
        citations: [
          { id: 'c1', shortTitle: 'BEST-Route', section: 'Method', chunkId: 'p1_c23' },
          { id: 'c2', shortTitle: 'BEST-Route', section: 'Experiments', chunkId: 'p1_c41' },
          { id: 'c3', shortTitle: 'UniRoute', section: 'Unified Framework', chunkId: 'p2_c18' },
          { id: 'c4', shortTitle: 'UniRoute', section: 'Results', chunkId: 'p2_c55' },
          { id: 'c5', shortTitle: 'BEST-Route', section: 'Results', chunkId: 'p1_c58' },
        ],
        retrievedChunks: [
          {
            chunkId: 'p1_c23',
            paperId: 'p1',
            paperTitle: 'BEST-Route',
            shortTitle: 'BEST-Route',
            section: 'Method',
            text: 'We define the Budget-Efficient Score with Tolerance (BEST) as S(q, m) = α · Acc(q, m) − (1−α) · Cost(q, m), where α is tuned per deployment context. The tolerance parameter ε = 0.05 allows routing to cheaper models when quality degradation remains within acceptable bounds.',
            score: 0.94,
            subQuery: 'What is the primary generation metric in BEST-Route?',
          },
          {
            chunkId: 'p1_c41',
            paperId: 'p1',
            paperTitle: 'BEST-Route',
            shortTitle: 'BEST-Route',
            section: 'Experiments',
            text: 'Across all benchmarks, BEST-Route achieves token budget adherence of 98.2% while maintaining accuracy within ε=0.05 of the strongest model baseline. This training-free approach requires no routing-specific fine-tuning.',
            score: 0.88,
            subQuery: 'What is the primary generation metric in BEST-Route?',
          },
          {
            chunkId: 'p2_c18',
            paperId: 'p2',
            paperTitle: 'UniRoute',
            shortTitle: 'UniRoute',
            section: 'Unified Framework',
            text: 'The Unified Routing Score (URS) is computed as URS(q) = w₁·Q̂(q) + w₂·Ê(q) where Q̂ and Ê are normalized quality and efficiency predictors. Weights w₁, w₂ are learned via joint optimization on routing supervision data.',
            score: 0.91,
            subQuery: 'What metric does UniRoute use for routing decisions?',
          },
          {
            chunkId: 'p2_c55',
            paperId: 'p2',
            paperTitle: 'UniRoute',
            shortTitle: 'UniRoute',
            section: 'Results',
            text: 'UniRoute achieves state-of-the-art routing efficiency on MMLU, GSM8K, and HumanEval, with 15% lower cost than single-model baselines at matched accuracy levels.',
            score: 0.82,
            subQuery: 'How do BEST-Route and UniRoute compare on evaluation benchmarks?',
          },
          {
            chunkId: 'p1_c58',
            paperId: 'p1',
            paperTitle: 'BEST-Route',
            shortTitle: 'BEST-Route',
            section: 'Results',
            text: 'On simple factoid queries (BEST score < 0.3), BEST-Route achieves 2.3× latency reduction by routing to haiku-tier models, with negligible accuracy loss (Δ = 0.012).',
            score: 0.79,
            subQuery: 'How do BEST-Route and UniRoute compare on evaluation benchmarks?',
          },
        ],
      },
      {
        id: 'm3',
        role: 'user',
        content: 'Does GraphRAG address multi-hop questions that BM25-based retrieval fails on?',
        timestamp: '2025-01-16T10:28:00Z',
      },
      {
        id: 'm4',
        role: 'assistant',
        content: `Yes — this is precisely GraphRAG's core contribution. The paper directly positions itself as a solution to multi-hop retrieval failures in BM25 and dense retrieval systems.

**Why BM25 fails on multi-hop questions** (per [From BM25 to Corrective RAG, §BM25 Era]): BM25 matches queries against individual chunks via lexical overlap. Multi-hop questions require chaining evidence across documents — e.g., "What evaluation metric does the paper cited by BEST-Route use?" — which requires traversing citation relationships that BM25 cannot represent.

**GraphRAG's approach** [GraphRAG, §Graph Construction]: The system extracts entities and relationships from all papers to build a knowledge graph. Retrieval then combines dense chunk similarity with graph traversal — starting from seed entities matching the query, then following edges to retrieve contextually connected chunks from other papers [GraphRAG, §Multi-Hop Retrieval].

**Experimental validation** [GraphRAG, §Experiments]: On multi-hop benchmarks, GraphRAG achieves 31% higher recall than BM25 hybrid retrieval. The improvement is most pronounced on 2-hop questions (47% gain) versus 3-hop (18% gain), suggesting graph traversal depth has diminishing returns.

**Caveat**: [From BM25 to Corrective RAG, §Open Problems] notes that graph construction quality depends heavily on entity extraction precision, and errors propagate across hops — a known limitation that GraphRAG acknowledges in its ablation study [GraphRAG, §Ablation Study].`,
        timestamp: '2025-01-16T10:28:45Z',
        confidence: 'high',
        processingMs: 6120,
        cacheHit: false,
        subQueries: [
          'Why does BM25 fail on multi-hop retrieval questions?',
          'How does GraphRAG construct the knowledge graph for retrieval?',
          'GraphRAG experimental results on multi-hop benchmarks',
          'Limitations of graph-based retrieval approaches',
        ],
        replanCount: 1,
        citations: [
          { id: 'c6', shortTitle: 'From BM25 to Corrective RAG', section: 'BM25 Era', chunkId: 'p4_c12' },
          { id: 'c7', shortTitle: 'GraphRAG', section: 'Graph Construction', chunkId: 'p5_c8' },
          { id: 'c8', shortTitle: 'GraphRAG', section: 'Multi-Hop Retrieval', chunkId: 'p5_c31' },
          { id: 'c9', shortTitle: 'GraphRAG', section: 'Experiments', chunkId: 'p5_c52' },
          { id: 'c10', shortTitle: 'From BM25 to Corrective RAG', section: 'Open Problems', chunkId: 'p4_c118' },
          { id: 'c11', shortTitle: 'GraphRAG', section: 'Ablation Study', chunkId: 'p5_c71' },
        ],
        retrievedChunks: [
          {
            chunkId: 'p4_c12',
            paperId: 'p4',
            paperTitle: 'From BM25 to Corrective RAG',
            shortTitle: 'From BM25 to Corrective RAG',
            section: 'BM25 Era',
            text: 'BM25 operates purely on lexical overlap between query tokens and document tokens. This design prevents it from reasoning over implicit relationships between documents — multi-hop questions that require connecting information across two or more passages are systematically out-of-scope for BM25.',
            score: 0.92,
            subQuery: 'Why does BM25 fail on multi-hop retrieval questions?',
          },
          {
            chunkId: 'p5_c8',
            paperId: 'p5',
            paperTitle: 'GraphRAG',
            shortTitle: 'GraphRAG',
            section: 'Graph Construction',
            text: 'For each ingested paper, we extract (entity, relation, entity) triples using a prompted LLM. Entities include methods, datasets, metrics, and authors. Relations include PROPOSES, EVALUATES-ON, CITES, and OUTPERFORMS. These are stored in a Neo4j graph with chunk-level provenance.',
            score: 0.95,
            subQuery: 'How does GraphRAG construct the knowledge graph for retrieval?',
          },
          {
            chunkId: 'p5_c31',
            paperId: 'p5',
            paperTitle: 'GraphRAG',
            shortTitle: 'GraphRAG',
            section: 'Multi-Hop Retrieval',
            text: 'Multi-hop retrieval proceeds in two stages: (1) seed retrieval identifies top-k chunks via dense similarity, extracting anchor entities; (2) graph traversal expands to neighboring entities up to 2 hops, collecting additional relevant chunks. The combined pool is reranked by a cross-encoder.',
            score: 0.89,
            subQuery: 'How does GraphRAG construct the knowledge graph for retrieval?',
          },
          {
            chunkId: 'p5_c52',
            paperId: 'p5',
            paperTitle: 'GraphRAG',
            shortTitle: 'GraphRAG',
            section: 'Experiments',
            text: 'On MuSiQue and 2WikiMultiHopQA, GraphRAG achieves 31% higher recall@5 vs. dense+BM25 hybrid. 2-hop questions benefit most (+47%), while 3-hop gains are smaller (+18%), consistent with noise accumulation at deeper traversal depths.',
            score: 0.87,
            subQuery: 'GraphRAG experimental results on multi-hop benchmarks',
          },
        ],
      },
    ],
  },
  {
    id: 'conv2',
    title: 'Memory Compression Strategies',
    createdAt: '2025-01-15T14:05:00Z',
    messages: [
      {
        id: 'm5',
        role: 'user',
        content: 'How does the conversational memory paper handle context compression for long sessions?',
        timestamp: '2025-01-15T14:05:00Z',
      },
      {
        id: 'm6',
        role: 'assistant',
        content: `The Conversational Memory paper proposes a hierarchical compression architecture with two distinct mechanisms operating at different time horizons [Conversational Memory, §Memory Architecture].

**Short-term memory** (current session): Sliding window over the last N turns, retained verbatim for immediate coherence. The window size N is calibrated per model context length [Conversational Memory, §Compression Strategies].

**Long-term memory** (across sessions): Older turns are summarized using a lightweight LM into *memory blocks* — structured summaries capturing entities, resolved references, and key facts. Memory blocks are retrieved via semantic similarity at query time, not prepended wholesale [Conversational Memory, §Retrieval-Based Memory].

**Compression ratio**: The paper reports 4× context reduction on a 20-turn benchmark with a BLEU degradation of only 2.1 points — a favorable trade-off [Conversational Memory, §Experiments]. The critical insight is that *verbatim recall is rarely needed* — what matters is preserving semantic content and entity state.

The authors note one important caveat: compression quality degrades for conversations with dense numerical data (e.g., code debugging sessions) where exact values must be preserved [Conversational Memory, §Compression Strategies].`,
        timestamp: '2025-01-15T14:05:38Z',
        confidence: 'high',
        processingMs: 3940,
        cacheHit: false,
        subQueries: [
          'How does Conversational Memory handle short-term context retention?',
          'What is the long-term memory compression strategy in the paper?',
          'Compression ratio and quality metrics in Conversational Memory paper',
        ],
        replanCount: 0,
        citations: [
          { id: 'c12', shortTitle: 'Conversational Memory', section: 'Memory Architecture', chunkId: 'p6_c5' },
          { id: 'c13', shortTitle: 'Conversational Memory', section: 'Compression Strategies', chunkId: 'p6_c22' },
          { id: 'c14', shortTitle: 'Conversational Memory', section: 'Retrieval-Based Memory', chunkId: 'p6_c38' },
          { id: 'c15', shortTitle: 'Conversational Memory', section: 'Experiments', chunkId: 'p6_c51' },
        ],
        retrievedChunks: [],
      },
    ],
  },
]
