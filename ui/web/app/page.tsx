import Link from 'next/link';
import { Brain, MessageSquare, Activity } from 'lucide-react';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-5xl w-full space-y-8 text-center">
        <div className="space-y-4">
          <Brain className="mx-auto h-24 w-24 text-indigo-600" />
          <h1 className="text-6xl font-bold tracking-tight text-gray-900">
            R-JEPA World Model
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Interactive reasoning assistant powered by a world model that understands
            conceptual relationships in latent space, inspired by Yann LeCun's JEPA vision.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-12">
          <Link
            href="/chat"
            className="group relative overflow-hidden rounded-xl bg-white p-8 shadow-lg transition-all hover:shadow-xl hover:scale-105"
          >
            <div className="flex flex-col items-center space-y-4">
              <MessageSquare className="h-16 w-16 text-indigo-600 group-hover:text-indigo-700" />
              <h2 className="text-2xl font-semibold text-gray-900">
                Chat Interface
              </h2>
              <p className="text-gray-600 text-center">
                Interact with the reasoning assistant. Toggle between different
                JEPA modes: rerank, nudge, and plan completion.
              </p>
            </div>
          </Link>

          <Link
            href="/jobs"
            className="group relative overflow-hidden rounded-xl bg-white p-8 shadow-lg transition-all hover:shadow-xl hover:scale-105"
          >
            <div className="flex flex-col items-center space-y-4">
              <Activity className="h-16 w-16 text-indigo-600 group-hover:text-indigo-700" />
              <h2 className="text-2xl font-semibold text-gray-900">
                Jobs & Monitoring
              </h2>
              <p className="text-gray-600 text-center">
                Monitor training jobs, dataset generation, and system metrics.
                Track the continuous improvement of the world model.
              </p>
            </div>
          </Link>
        </div>

        <div className="mt-12 p-6 bg-white rounded-xl shadow-md">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            What is R-JEPA?
          </h3>
          <p className="text-gray-700 text-left max-w-3xl mx-auto">
            R-JEPA (Reasoning Joint Embedding Predictive Architecture) is a world model
            for textual reasoning. Like V-JEPA learns visual world physics from video,
            R-JEPA learns conceptual reasoning patterns from latent representations.
            It predicts masked reasoning steps in latent space, enabling re-ranking,
            correction, and completion of chains of thought.
          </p>
        </div>

        <footer className="mt-12 text-sm text-gray-500">
          <p>Powered by Qwen3-8B + R-JEPA | Version 0.1.0 (Phase 9 MVP)</p>
        </footer>
      </div>
    </main>
  );
}
