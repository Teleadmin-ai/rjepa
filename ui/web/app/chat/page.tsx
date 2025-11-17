'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Brain, Send, ThumbsUp, ThumbsDown, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { sendChatMessage, submitFeedback, type JepaMode, type ChatResponse } from '@/lib/api';
import { cn } from '@/lib/utils';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  jepa_data?: ChatResponse;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [jepaMode, setJepaMode] = useState<JepaMode>('rerank');
  const [numSamples, setNumSamples] = useState(4);
  const [temperature, setTemperature] = useState(0.7);

  const chatMutation = useMutation({
    mutationFn: sendChatMessage,
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: 'assistant',
          content: data.answer,
          jepa_data: data,
        },
      ]);
      setInput('');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);

    chatMutation.mutate({
      prompt: input.trim(),
      mode: jepaMode,
      num_samples: numSamples,
      temperature,
    });
  };

  const handleFeedback = async (messageId: string, feedback: 'thumbs_up' | 'thumbs_down') => {
    const message = messages.find((m) => m.id === messageId);
    if (!message || !message.jepa_data) return;

    const userMessage = messages.find(
      (m) => m.role === 'user' && messages.indexOf(m) === messages.indexOf(message) - 1
    );

    if (!userMessage) return;

    try {
      await submitFeedback({
        session_id: messageId,
        prompt: userMessage.content,
        answer: message.content,
        feedback,
        jepa_score: message.jepa_data.jepa_score,
      });
      alert(`Feedback sent: ${feedback}`);
    } catch (error) {
      console.error('Failed to send feedback:', error);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Brain className="h-8 w-8 text-indigo-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">R-JEPA Chat</h1>
              <p className="text-sm text-gray-500">Reasoning Assistant powered by World Model</p>
            </div>
          </div>

          {/* Mode Selector */}
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">JEPA Mode:</span>
            {(['off', 'rerank', 'nudge', 'plan'] as JepaMode[]).map((mode) => (
              <Button
                key={mode}
                variant={jepaMode === mode ? 'default' : 'outline'}
                size="sm"
                onClick={() => setJepaMode(mode)}
                className="capitalize"
              >
                {mode}
              </Button>
            ))}
          </div>
        </div>
      </header>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <Brain className="h-16 w-16 text-gray-400 mx-auto mb-4" />
              <h2 className="text-xl font-semibold text-gray-700 mb-2">
                Start a conversation
              </h2>
              <p className="text-gray-500">
                Ask a question and I'll help you reason through it.
              </p>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                'flex',
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              {message.role === 'user' ? (
                <div className="bg-indigo-600 text-white rounded-lg px-4 py-3 max-w-2xl">
                  <p className="text-sm">{message.content}</p>
                </div>
              ) : (
                <Card className="max-w-3xl w-full">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">Assistant</CardTitle>
                      {message.jepa_data && (
                        <Badge variant="secondary">
                          Mode: {message.jepa_data.mode}
                        </Badge>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Main Answer */}
                    <div className="prose prose-sm max-w-none">
                      <p className="text-gray-800 whitespace-pre-wrap">{message.content}</p>
                    </div>

                    {/* Reasoning Steps */}
                    {message.jepa_data && message.jepa_data.steps.length > 0 && (
                      <details className="bg-gray-50 rounded-md p-4">
                        <summary className="cursor-pointer font-medium text-sm text-gray-700">
                          View Reasoning Steps ({message.jepa_data.steps.length} steps)
                        </summary>
                        <div className="mt-3 space-y-2">
                          {message.jepa_data.steps.map((step, idx) => (
                            <div key={idx} className="text-sm text-gray-600 pl-4 border-l-2 border-indigo-300">
                              {step}
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    {/* JEPA Details */}
                    {message.jepa_data && message.jepa_data.mode !== 'off' && (
                      <details className="bg-blue-50 rounded-md p-4">
                        <summary className="cursor-pointer font-medium text-sm text-blue-900">
                          JEPA Details
                        </summary>
                        <div className="mt-3 space-y-2 text-sm">
                          {message.jepa_data.jepa_score !== null && (
                            <div className="flex justify-between">
                              <span className="text-gray-600">JEPA Score:</span>
                              <span className="font-mono text-gray-900">
                                {message.jepa_data.jepa_score.toFixed(4)}
                              </span>
                            </div>
                          )}
                          {message.jepa_data.candidates && (
                            <div>
                              <span className="text-gray-600 block mb-2">
                                Candidates ({message.jepa_data.candidates.length}):
                              </span>
                              <div className="space-y-1">
                                {message.jepa_data.candidates.map((cand, idx) => (
                                  <div
                                    key={idx}
                                    className="text-xs bg-white rounded p-2 border border-blue-200"
                                  >
                                    <div className="flex justify-between mb-1">
                                      <Badge variant={idx === 0 ? 'default' : 'outline'} className="text-xs">
                                        {idx === 0 ? 'Selected' : `Candidate ${idx + 1}`}
                                      </Badge>
                                      <span className="font-mono">
                                        JEPA: {cand.jepa_loss.toFixed(3)}
                                      </span>
                                    </div>
                                    <p className="text-gray-600 truncate">{cand.text}</p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                          {Object.keys(message.jepa_data.metadata).length > 0 && (
                            <div className="text-xs text-gray-500 font-mono">
                              {JSON.stringify(message.jepa_data.metadata, null, 2)}
                            </div>
                          )}
                        </div>
                      </details>
                    )}

                    {/* Feedback Buttons */}
                    <div className="flex items-center space-x-2 pt-2 border-t">
                      <span className="text-xs text-gray-500">Was this helpful?</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleFeedback(message.id, 'thumbs_up')}
                      >
                        <ThumbsUp className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleFeedback(message.id, 'thumbs_down')}
                      >
                        <ThumbsDown className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          ))}

          {chatMutation.isPending && (
            <div className="flex justify-start">
              <Card className="max-w-3xl w-full">
                <CardContent className="p-6">
                  <div className="flex items-center space-x-3">
                    <Loader2 className="h-5 w-5 animate-spin text-indigo-600" />
                    <span className="text-sm text-gray-600">
                      R-JEPA is thinking... (mode: {jepaMode})
                    </span>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 px-6 py-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex space-x-4">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question (e.g., Solve 2x + 5 = 13)..."
              className="flex-1"
              rows={3}
              disabled={chatMutation.isPending}
            />
            <Button
              type="submit"
              disabled={chatMutation.isPending || !input.trim()}
              className="px-8"
            >
              <Send className="h-5 w-5" />
            </Button>
          </div>

          {/* Advanced Options */}
          {jepaMode === 'rerank' && (
            <div className="mt-3 flex items-center space-x-4 text-sm">
              <label className="flex items-center space-x-2">
                <span className="text-gray-600">Candidates:</span>
                <input
                  type="number"
                  value={numSamples}
                  onChange={(e) => setNumSamples(Number(e.target.value))}
                  min={2}
                  max={10}
                  className="w-16 px-2 py-1 border rounded"
                />
              </label>
              <label className="flex items-center space-x-2">
                <span className="text-gray-600">Temperature:</span>
                <input
                  type="number"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  min={0}
                  max={2}
                  step={0.1}
                  className="w-16 px-2 py-1 border rounded"
                />
              </label>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}
