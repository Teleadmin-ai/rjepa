'use client';

import { useQuery } from '@tanstack/react-query';
import { Activity, Clock, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { getJobs, type JobStatus } from '@/lib/api';

function getStatusIcon(status: string) {
  switch (status) {
    case 'running':
      return <Loader2 className="h-5 w-5 animate-spin text-blue-500" />;
    case 'success':
      return <CheckCircle className="h-5 w-5 text-green-500" />;
    case 'failed':
      return <XCircle className="h-5 w-5 text-red-500" />;
    case 'queued':
      return <Clock className="h-5 w-5 text-gray-400" />;
    default:
      return <Activity className="h-5 w-5 text-gray-400" />;
  }
}

function getStatusBadge(status: string) {
  switch (status) {
    case 'running':
      return <Badge>Running</Badge>;
    case 'success':
      return <Badge variant="secondary">Success</Badge>;
    case 'failed':
      return <Badge variant="destructive">Failed</Badge>;
    case 'queued':
      return <Badge variant="outline">Queued</Badge>;
    default:
      return <Badge variant="outline">{status}</Badge>;
  }
}

function formatJobType(jobType: string): string {
  return jobType
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function JobCard({ job }: { job: JobStatus }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {getStatusIcon(job.status)}
            <div>
              <CardTitle className="text-lg">{formatJobType(job.job_type)}</CardTitle>
              <CardDescription className="text-xs font-mono">{job.job_id}</CardDescription>
            </div>
          </div>
          {getStatusBadge(job.status)}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Progress */}
        {job.status === 'running' && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Progress</span>
              <span className="font-medium">{Math.round(job.progress * 100)}%</span>
            </div>
            <Progress value={job.progress * 100} />
          </div>
        )}

        {/* Timestamps */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500 block text-xs">Created</span>
            <span className="font-mono text-xs">{new Date(job.created_at).toLocaleString()}</span>
          </div>
          <div>
            <span className="text-gray-500 block text-xs">Updated</span>
            <span className="font-mono text-xs">{new Date(job.updated_at).toLocaleString()}</span>
          </div>
        </div>

        {/* Metadata */}
        {Object.keys(job.metadata).length > 0 && (
          <details className="bg-gray-50 rounded-md p-3">
            <summary className="cursor-pointer text-sm font-medium text-gray-700">
              Metadata
            </summary>
            <div className="mt-2 space-y-1">
              {Object.entries(job.metadata).map(([key, value]) => (
                <div key={key} className="flex justify-between text-xs">
                  <span className="text-gray-600">{key}:</span>
                  <span className="font-mono text-gray-900">{String(value)}</span>
                </div>
              ))}
            </div>
          </details>
        )}
      </CardContent>
    </Card>
  );
}

export default function JobsPage() {
  const { data: jobs, isLoading, error, refetch } = useQuery({
    queryKey: ['jobs'],
    queryFn: getJobs,
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Activity className="h-8 w-8 text-indigo-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Jobs & Monitoring</h1>
                <p className="text-sm text-gray-500">
                  Track training jobs, dataset generation, and system tasks
                </p>
              </div>
            </div>

            <button
              onClick={() => refetch()}
              className="px-4 py-2 text-sm font-medium text-indigo-600 hover:bg-indigo-50 rounded-md transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>
      </header>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-6 py-8">
        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-indigo-600" />
            <span className="ml-3 text-gray-600">Loading jobs...</span>
          </div>
        )}

        {error && (
          <Card className="border-red-200 bg-red-50">
            <CardContent className="p-6">
              <div className="flex items-center space-x-3">
                <XCircle className="h-6 w-6 text-red-600" />
                <div>
                  <p className="font-medium text-red-900">Failed to load jobs</p>
                  <p className="text-sm text-red-700">
                    {error instanceof Error ? error.message : 'Unknown error'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {jobs && jobs.length === 0 && (
          <div className="text-center py-12">
            <Activity className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-700 mb-2">No jobs running</h2>
            <p className="text-gray-500">
              Jobs will appear here when they are created
            </p>
          </div>
        )}

        {jobs && jobs.length > 0 && (
          <>
            {/* Stats Summary */}
            <div className="grid grid-cols-4 gap-4 mb-8">
              {(['queued', 'running', 'success', 'failed'] as const).map((status) => {
                const count = jobs.filter((j) => j.status === status).length;
                return (
                  <Card key={status}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-2xl font-bold">{count}</p>
                          <p className="text-xs text-gray-500 capitalize">{status}</p>
                        </div>
                        {getStatusIcon(status)}
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>

            {/* Jobs List */}
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-gray-900">
                All Jobs ({jobs.length})
              </h2>
              {jobs.map((job) => (
                <JobCard key={job.job_id} job={job} />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
