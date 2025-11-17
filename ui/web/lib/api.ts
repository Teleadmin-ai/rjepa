export type JepaMode = 'off' | 'rerank' | 'nudge' | 'plan';

export interface ChatRequest {
  prompt: string;
  mode: JepaMode;
  num_samples?: number;
  temperature?: number;
  domain_id?: number | null;
}

export interface ChatResponse {
  answer: string;
  steps: string[];
  jepa_score: number | null;
  mode: JepaMode;
  candidates?: Array<{
    text: string;
    score: number;
    jepa_loss: number;
  }>;
  metadata: Record<string, any>;
}

export interface FeedbackRequest {
  session_id: string;
  prompt: string;
  answer: string;
  feedback: 'thumbs_up' | 'thumbs_down' | string;
  jepa_score: number | null;
}

export interface JobStatus {
  job_id: string;
  job_type: string;
  status: 'queued' | 'running' | 'success' | 'failed';
  progress: number;
  created_at: string;
  updated_at: string;
  metadata: Record<string, any>;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8300';

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Chat API error: ${response.statusText}`);
  }

  return response.json();
}

export async function submitFeedback(request: FeedbackRequest): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/feedback`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Feedback API error: ${response.statusText}`);
  }

  return response.json();
}

export async function getJobs(): Promise<JobStatus[]> {
  const response = await fetch(`${API_BASE_URL}/api/jobs`);

  if (!response.ok) {
    throw new Error(`Jobs API error: ${response.statusText}`);
  }

  return response.json();
}
