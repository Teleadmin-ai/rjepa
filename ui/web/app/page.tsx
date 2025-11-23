import Link from 'next/link';
import { Brain, MessageSquare, Activity, Code2, Cloud, Github, BookOpen, Shield } from 'lucide-react';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center p-8 md:p-24 bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-6xl w-full space-y-12">
        {/* Hero Section */}
        <div className="text-center space-y-4">
          <Brain className="mx-auto h-24 w-24 text-indigo-600" />
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight text-gray-900">
            R-JEPA World Model
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            A world model for textual reasoning that understands conceptual relationships
            in latent space, inspired by Yann LeCun's JEPA vision.
          </p>
          <div className="flex items-center justify-center gap-2 text-sm text-indigo-600 font-medium">
            <Shield className="h-4 w-4" />
            <span>MIT Licensed (V-JEPA 2) | Commercial Use OK</span>
          </div>
        </div>

        {/* Main Features */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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

        {/* Research & Service Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
          {/* Research (Open Source) */}
          <div className="bg-white rounded-xl shadow-lg p-8 border-l-4 border-green-500">
            <div className="flex items-center gap-3 mb-4">
              <Code2 className="h-8 w-8 text-green-600" />
              <h3 className="text-2xl font-bold text-gray-900">Research</h3>
              <span className="bg-green-100 text-green-800 text-xs font-medium px-2.5 py-0.5 rounded">
                Open Source
              </span>
            </div>
            <p className="text-gray-700 mb-6">
              Full R-JEPA implementation open-sourced under MIT license. Train your own
              world model for reasoning on any LLM.
            </p>
            <ul className="space-y-3 text-gray-600 mb-6">
              <li className="flex items-start gap-2">
                <Github className="h-5 w-5 text-gray-500 mt-0.5" />
                <span>Complete architecture: Encoder, Predictor, EMA</span>
              </li>
              <li className="flex items-start gap-2">
                <BookOpen className="h-5 w-5 text-gray-500 mt-0.5" />
                <span>Training pipeline with Prefect orchestration</span>
              </li>
              <li className="flex items-start gap-2">
                <Code2 className="h-5 w-5 text-gray-500 mt-0.5" />
                <span>Multi-LLM support (Qwen, Llama, Mistral...)</span>
              </li>
            </ul>
            <a
              href="https://github.com/Teleadmin-ai/rjepa"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 bg-gray-900 text-white px-4 py-2 rounded-lg hover:bg-gray-800 transition-colors"
            >
              <Github className="h-5 w-5" />
              View on GitHub
            </a>
          </div>

          {/* Service (Commercial) */}
          <div className="bg-white rounded-xl shadow-lg p-8 border-l-4 border-indigo-500">
            <div className="flex items-center gap-3 mb-4">
              <Cloud className="h-8 w-8 text-indigo-600" />
              <h3 className="text-2xl font-bold text-gray-900">Service</h3>
              <span className="bg-indigo-100 text-indigo-800 text-xs font-medium px-2.5 py-0.5 rounded">
                API Cloud
              </span>
            </div>
            <p className="text-gray-700 mb-6">
              Pre-trained R-JEPA models via API. Your data stays local - only latent
              vectors are transmitted. GDPR/HIPAA friendly.
            </p>
            <ul className="space-y-3 text-gray-600 mb-6">
              <li className="flex items-start gap-2">
                <Shield className="h-5 w-5 text-indigo-500 mt-0.5" />
                <span>Privacy-first: Text never leaves your infrastructure</span>
              </li>
              <li className="flex items-start gap-2">
                <Cloud className="h-5 w-5 text-indigo-500 mt-0.5" />
                <span>Pre-trained checkpoints for instant deployment</span>
              </li>
              <li className="flex items-start gap-2">
                <Activity className="h-5 w-5 text-indigo-500 mt-0.5" />
                <span>Pay per token, contextual memory support</span>
              </li>
            </ul>
            <a
              href="mailto:commercial@teleadmin.net?subject=R-JEPA%20API%20Access"
              className="inline-flex items-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
            >
              <Cloud className="h-5 w-5" />
              Request API Access
            </a>
          </div>
        </div>

        {/* What is R-JEPA */}
        <div className="p-8 bg-white rounded-xl shadow-md">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">
            What is R-JEPA?
          </h3>
          <div className="prose prose-gray max-w-none text-gray-700">
            <p className="mb-4">
              <strong>R-JEPA (Reasoning Joint Embedding Predictive Architecture)</strong> is a world model
              for textual reasoning. Like V-JEPA learns visual world physics from video,
              R-JEPA learns conceptual reasoning patterns from latent representations.
            </p>
            <p className="mb-4">
              It predicts masked reasoning steps in latent space, enabling:
            </p>
            <ul className="list-disc pl-6 space-y-2">
              <li><strong>Re-ranking</strong>: Choose the best chain of thought from multiple candidates</li>
              <li><strong>Nudge</strong>: Gently correct reasoning that's going off-track</li>
              <li><strong>Plan</strong>: Complete missing steps in partial reasoning</li>
            </ul>
            <p className="mt-4 text-sm text-gray-500">
              Based on Meta AI's V-JEPA 2 architecture (MIT license) - adapted from visual patches to reasoning steps.
            </p>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-sm text-gray-500 space-y-2">
          <p>Powered by Qwen3-8B + R-JEPA | V-JEPA 2 (MIT License)</p>
          <p className="text-xs">
            Version 0.2.0 | Phase 26 Complete
          </p>
          <div className="flex items-center justify-center gap-4 mt-2">
            <a
              href="https://github.com/Teleadmin-ai/rjepa"
              className="text-gray-400 hover:text-gray-600"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Github className="h-5 w-5" />
            </a>
            <span className="text-gray-300">|</span>
            <a
              href="https://teleadmin.net"
              className="text-gray-400 hover:text-gray-600"
              target="_blank"
              rel="noopener noreferrer"
            >
              Teleadmin.net
            </a>
          </div>
        </footer>
      </div>
    </main>
  );
}
