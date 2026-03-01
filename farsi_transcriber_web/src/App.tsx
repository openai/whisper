import { useState, useRef, useEffect, useMemo } from 'react';
import {
  FileAudio,
  Upload,
  Moon,
  Sun,
  Search,
  Copy,
  X,
  CheckCircle2,
  Clock,
  Loader2,
  Download
} from 'lucide-react';
import { Resizable } from 're-resizable';
import { Toaster, toast } from 'sonner';
import Button from './components/Button';
import Progress from './components/Progress';
import Input from './components/Input';
import Select from './components/Select';

interface FileItem {
  id: string; // This is the internal UI ID
  jobId?: string; // This is the backend Job ID
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress?: number;
  transcription?: TranscriptionSegment[];
  file?: File;
  fullText?: string;
}

interface TranscriptionSegment {
  start: string;
  end: string;
  text: string;
  speaker?: string;
}

interface HighlightedSegment extends TranscriptionSegment {
  highlightedHtml: string;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

export default function App() {
  const [fileQueue, setFileQueue] = useState<FileItem[]>([]);
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const [isDark, setIsDark] = useState(false);
  const [windowSize, setWindowSize] = useState({ width: 1100, height: 700 });
  const [searchQuery, setSearchQuery] = useState('');
  const [exportFormat, setExportFormat] = useState('txt');
  const [modelSize, setModelSize] = useState('medium');
  const [isDiarizationEnabled, setIsDiarizationEnabled] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Theme colors
  const theme = {
    bg: isDark ? '#1a1a1a' : '#f5f5f5',
    cardBg: isDark ? '#2d2d2d' : '#ffffff',
    inputBg: isDark ? '#3a3a3a' : '#f9f9f9',
    border: isDark ? '#4a4a4a' : '#d0d0d0',
    text: isDark ? '#e0e0e0' : '#333333',
    textSecondary: isDark ? '#a0a0a0' : '#666666',
    progressBg: isDark ? '#404040' : '#e0e0e0',
    sidebarBg: isDark ? '#252525' : '#fafafa',
    hoverBg: isDark ? '#3a3a3a' : '#f0f0f0',
    selectedBg: isDark ? '#4a4a4a' : '#e8f5e9',
  };

  const handleAddFiles = () => {
    fileInputRef.current?.click();
  };

  const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const newFiles: FileItem[] = Array.from(event.target.files).map(file => ({
        id: Date.now().toString() + Math.random().toString(),
        name: file.name,
        status: 'pending',
        file: file
      }));

      setFileQueue(prev => [...prev, ...newFiles]);
      if (!selectedFileId && newFiles.length > 0) {
        setSelectedFileId(newFiles[0].id);
      }
      toast.success(`${newFiles.length} file(s) added to queue`);
    }
    // Reset input
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleRemoveFile = (id: string) => {
    setFileQueue(fileQueue.filter(f => f.id !== id));
    if (selectedFileId === id) {
      setSelectedFileId(fileQueue[0]?.id || null);
    }
    toast.info('File removed from queue');
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const newFiles: FileItem[] = Array.from(e.dataTransfer.files).map(file => ({
        id: Date.now().toString() + Math.random().toString(),
        name: file.name,
        status: 'pending',
        file: file
      }));

      setFileQueue(prev => [...prev, ...newFiles]);
      if (!selectedFileId && newFiles.length > 0) {
        setSelectedFileId(newFiles[0].id);
      }
      toast.success(`${newFiles.length} file(s) added to queue`);
    }
  };

  const pollJobStatus = async (uiId: string, jobId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
        const data = await response.json();

        if (response.status === 404) {
          clearInterval(pollInterval);
          setFileQueue(prev => prev.map(f => f.id === uiId ? { ...f, status: 'error' } : f));
          toast.error('Job not found');
          return;
        }

        if (data.status === 'completed') {
          clearInterval(pollInterval);
          // Fetch result
          const resultResponse = await fetch(`${API_BASE_URL}/jobs/${jobId}/result`);
          const resultData = await resultResponse.json();

          setFileQueue(prev => prev.map(f => {
            if (f.id === uiId) {
              return {
                ...f,
                status: 'completed',
                progress: 100,
                transcription: resultData.segments,
                fullText: resultData.full_text || resultData.text
              };
            }
            return f;
          }));
          toast.success('Transcription completed!');
        } else if (data.status === 'error') {
          clearInterval(pollInterval);
          setFileQueue(prev => prev.map(f => f.id === uiId ? { ...f, status: 'error' } : f));
          toast.error(`Transcription failed: ${data.error}`);
        } else {
          // Processing or pending
          setFileQueue(prev => prev.map(f => {
            if (f.id === uiId) {
              return {
                ...f,
                status: data.status,
                progress: data.progress || (data.status === 'processing' ? 50 : 0) // Fake progress if API doesn't provide
              };
            }
            return f;
          }));
        }

      } catch (error) {
        console.error("Polling error", error);
        // Don't stop polling on transient network errors immediately, but maybe implementing a retry limit is good.
      }
    }, 2000);
  };

  const handleTranscribe = async () => {
    if (!selectedFileId) return;

    const fileItem = fileQueue.find(f => f.id === selectedFileId);
    if (!fileItem || !fileItem.file) return;

    // Update status to uploading/processing
    setFileQueue(prev => prev.map(f => f.id === selectedFileId ? { ...f, status: 'processing', progress: 0 } : f));

    const formData = new FormData();
    formData.append('file', fileItem.file);
    formData.append('language', 'fa');
    formData.append('model', modelSize);
    formData.append('diarization', isDiarizationEnabled.toString());

    try {
      const response = await fetch(`${API_BASE_URL}/jobs`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const data = await response.json();
      const jobId = data.job_id;

      // Update with job ID and start polling
      setFileQueue(prev => prev.map(f => f.id === selectedFileId ? { ...f, jobId: jobId } : f));

      pollJobStatus(selectedFileId, jobId);

    } catch (error: any) {
      setFileQueue(prev => prev.map(f => f.id === selectedFileId ? { ...f, status: 'error' } : f));
      toast.error(error.message || 'Failed to start transcription');
    }
  };

  const handleCopySegment = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard');
  };

  const handleExport = async () => {
    const selectedFile = fileQueue.find(f => f.id === selectedFileId);
    if (!selectedFile?.jobId || selectedFile.status !== 'completed') {
      toast.error('No completed transcription to export');
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/export/${selectedFile.jobId}?format=${exportFormat}`);
      if (!response.ok) throw new Error('Export failed');

      const data = await response.json();

      // Create a blob and download
      const blob = new Blob([data.content], { type: data.mime_type });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedFile.name.split('.')[0]}.${exportFormat}`; // Simple rename
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast.success(`Exported as ${exportFormat.toUpperCase()}`);
    } catch (error) {
      toast.error('Failed to export file');
    }
  };

  const handleClearAll = () => {
    setFileQueue([]);
    setSelectedFileId(null);
    setSearchQuery('');
    toast.info('All files cleared');
  };

  const selectedFile = fileQueue.find(f => f.id === selectedFileId);
  const currentTranscription = selectedFile?.transcription || [];

  // Memoize search regex for performance
  const searchRegex = useMemo(() => {
    if (!searchQuery) return null;
    try {
      // Escape special regex characters to prevent crashes
      const escapedQuery = searchQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      return new RegExp(`(${escapedQuery})`, 'gi');
    } catch (e) {
      return null;
    }
  }, [searchQuery]);

  // Filter transcription segments based on search query
  const filteredSegments = useMemo(() => {
    if (!searchQuery) return currentTranscription;
    return currentTranscription.filter(seg =>
      seg.text.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [currentTranscription, searchQuery]);

  // Apply highlighting to filtered segments
  const filteredTranscription = useMemo((): (TranscriptionSegment | HighlightedSegment)[] => {
    if (!searchQuery || !searchRegex) {
      return filteredSegments;
    }

    return filteredSegments.map(seg => ({
      ...seg,
      highlightedHtml: seg.text.split(searchRegex).map((part) =>
        part.toLowerCase() === searchQuery.toLowerCase()
          ? `<mark style="background-color: ${isDark ? '#4CAF50' : '#FFEB3B'}; color: ${isDark ? '#000' : '#000'}; padding: 2px 4px; border-radius: 2px;">${part}</mark>`
          : part
      ).join('')
    }));
  }, [filteredSegments, searchQuery, searchRegex, isDark]);

  const getStatusIcon = (status: FileItem['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-4 h-4 text-green-500" />;
      case 'processing':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'error':
        return <X className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4" style={{ color: theme.textSecondary }} />;
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-8" style={{ backgroundColor: theme.bg }}>
      <Toaster theme={isDark ? 'dark' : 'light'} position="top-right" />

      {/* Hidden File Input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={onFileChange}
        style={{ display: 'none' }}
        accept="audio/*,video/*"
        multiple
      />

      <Resizable
        size={windowSize}
        onResizeStop={(_e, _direction, _ref, d) => {
          setWindowSize({
            width: windowSize.width + d.width,
            height: windowSize.height + d.height,
          });
        }}
        minWidth={900}
        minHeight={600}
        className="rounded-lg shadow-2xl overflow-hidden"
        style={{
          backgroundColor: theme.cardBg,
          border: `2px solid ${theme.border}`,
        }}
        handleStyles={{
          right: { cursor: 'ew-resize' },
          bottom: { cursor: 'ns-resize' },
          bottomRight: { cursor: 'nwse-resize' },
        }}
      >
        <div className="flex h-full">
          {/* Left Sidebar - File Queue */}
          <div
            className={`w-64 border-r flex flex-col overflow-hidden transition-colors`}
            style={{
              borderColor: theme.border,
              backgroundColor: isDragging ? (isDark ? '#2a2a2a' : '#f0f9ff') : theme.sidebarBg,
              borderStyle: isDragging ? 'dashed' : 'solid',
              borderWidth: isDragging ? '2px' : '0 1px 0 0',
              borderColor: isDragging ? '#3b82f6' : theme.border
            }}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="p-4 border-b" style={{ borderColor: theme.border }}>
              <h3 className="mb-3 font-semibold" style={{ color: theme.text }}>
                File Queue
              </h3>
              <Button
                onClick={handleAddFiles}
                className="w-full bg-green-500 hover:bg-green-600 text-white"
              >
                <Upload className="w-4 h-4 mr-2" />
                Add Files
              </Button>
            </div>

            <div className="flex-1 overflow-auto p-2">
              {fileQueue.length === 0 ? (
                <p className="text-center text-xs p-4" style={{ color: theme.textSecondary }}>
                  No files in queue
                </p>
              ) : (
                fileQueue.map((file) => (
                  <div
                    key={file.id}
                    className="mb-2 p-3 rounded-lg cursor-pointer transition-colors border"
                    style={{
                      backgroundColor: selectedFileId === file.id ? theme.selectedBg : theme.cardBg,
                      borderColor: selectedFileId === file.id ? '#4CAF50' : theme.border,
                    }}
                    onClick={() => setSelectedFileId(file.id)}
                  >
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div className="flex items-center gap-2 flex-1 min-w-0">
                        {getStatusIcon(file.status)}
                        <span className="text-xs truncate" style={{ color: theme.text }}>
                          {file.name}
                        </span>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveFile(file.id);
                        }}
                        className="hover:opacity-70"
                      >
                        <X className="w-3 h-3" style={{ color: theme.textSecondary }} />
                      </button>
                    </div>
                    {file.status === 'processing' && (
                      <div className="space-y-1">
                        <Progress value={file.progress || 0} />
                        <p className="text-xs" style={{ color: theme.textSecondary }}>
                          {file.progress}%
                        </p>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Main Content Area */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Header */}
            <div
              className="p-5 border-b flex items-center justify-between"
              style={{ borderColor: theme.border }}
            >
              <div className="flex items-center gap-3">
                <h1 style={{ color: theme.text }} className="text-lg font-semibold">
                  Farsi Audio/Video Transcriber
                </h1>
                <span className="text-xs" style={{ color: theme.textSecondary }}>
                  {windowSize.width}×{windowSize.height}
                </span>
              </div>
              <Button
                onClick={() => setIsDark(!isDark)}
                variant="outline"
                style={{ borderColor: theme.border, backgroundColor: theme.cardBg }}
              >
                {isDark ? (
                  <Sun className="w-4 h-4" style={{ color: theme.text }} />
                ) : (
                  <Moon className="w-4 h-4" style={{ color: theme.text }} />
                )}
              </Button>
            </div>

            <div className="flex-1 flex flex-col p-5 overflow-hidden">
              {/* File Info & Actions */}
              <div
                className="mb-4 p-4 rounded-lg border"
                style={{ backgroundColor: theme.inputBg, borderColor: theme.border }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <FileAudio className="w-5 h-5" style={{ color: theme.textSecondary }} />
                    <div>
                      <p className="text-sm" style={{ color: theme.text }}>
                        {selectedFile ? selectedFile.name : 'No file selected'}
                      </p>
                      {selectedFile?.status === 'processing' && (
                        <p className="text-xs" style={{ color: theme.textSecondary }}>
                          Processing... {selectedFile.progress}%
                        </p>
                      )}
                      {selectedFile?.status === 'completed' && (
                        <p className="text-xs text-green-500">Completed</p>
                      )}
                    </div>
                  </div>
                  <Button
                    onClick={handleTranscribe}
                    disabled={!selectedFile || selectedFile.status === 'processing' || selectedFile.status === 'completed'}
                    className="bg-green-500 hover:bg-green-600 text-white disabled:bg-gray-400 disabled:cursor-not-allowed"
                  >
                    {selectedFile?.status === 'processing' ? 'Transcribing...' : 'Transcribe'}
                  </Button>
                </div>
              </div>

              {/* Model Selection */}
              <div className="mb-4 flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <label style={{ color: theme.text }} className="text-sm font-medium">Model Size:</label>
                  <Select
                    value={modelSize}
                    onChange={(e) => setModelSize(e.target.value)}
                    style={{ width: '150px' }}
                  >
                    <option value="tiny">Tiny (Fastest)</option>
                    <option value="base">Base</option>
                    <option value="small">Small</option>
                    <option value="medium">Medium (Balanced)</option>
                    <option value="large">Large (Best Accuracy)</option>
                  </Select>
                </div>
                <p className="text-xs" style={{ color: theme.textSecondary }}>
                  Larger models are more accurate but take longer to process.
                </p>
              </div>

              {/* Diarization Toggle */}
              <div className="mb-4 flex items-center gap-2">
                <input
                  type="checkbox"
                  id="diarization"
                  checked={isDiarizationEnabled}
                  onChange={(e) => setIsDiarizationEnabled(e.target.checked)}
                  className="w-4 h-4"
                />
                <label htmlFor="diarization" style={{ color: theme.text }} className="text-sm font-medium cursor-pointer">
                  Enable Speaker Diarization (Identify Speakers)
                </label>
              </div>

              {/* Search & Export Controls */}
              {selectedFile?.transcription && (
                <div className="mb-4 flex gap-2">
                  <div className="flex-1 relative">
                    <Search
                      className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2"
                      style={{ color: theme.textSecondary }}
                    />
                    <Input
                      placeholder="Search in transcription..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      style={{
                        backgroundColor: theme.inputBg,
                        borderColor: theme.border,
                        color: theme.text,
                        paddingLeft: '2.25rem',
                      }}
                    />
                  </div>
                  <Select
                    value={exportFormat}
                    onChange={(e) => setExportFormat(e.target.value as 'txt' | 'docx' | 'pdf' | 'srt')}
                  >
                    <option value="txt">TXT</option>
                    <option value="json">JSON</option>
                    <option value="srt">SRT</option>
                    <option value="vtt">VTT</option>
                    <option value="tsv">TSV</option>
                  </Select>
                  <Button
                    onClick={handleExport}
                    variant="outline"
                    style={{ borderColor: theme.border, backgroundColor: theme.cardBg, color: theme.text }}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Export
                  </Button>
                </div>
              )}

              {/* Transcription Results */}
              <div className="flex-1 flex flex-col min-h-0">
                <div className="flex items-center justify-between mb-2">
                  <label style={{ color: theme.text }} className="text-sm font-medium">
                    Transcription Results:
                  </label>
                  {searchQuery && (
                    <span className="text-xs" style={{ color: theme.textSecondary }}>
                      {filteredTranscription.length} results found
                    </span>
                  )}
                </div>

                <div
                  className="flex-1 rounded-lg border p-4 overflow-auto"
                  style={{ backgroundColor: theme.cardBg, borderColor: theme.border }}
                >
                  {currentTranscription.length === 0 ? (
                    <p className="text-center" style={{ color: theme.textSecondary }}>
                      Transcription results will appear here...
                    </p>
                  ) : (
                    <div className="space-y-3">
                      {filteredTranscription.map((segment, index) => (
                        <div
                          key={index}
                          className="p-3 rounded-md border group hover:shadow-sm transition-shadow"
                          style={{
                            backgroundColor: theme.inputBg,
                            borderColor: theme.border,
                          }}
                        >
                          <div className="flex items-start justify-between gap-3 mb-2">
                            <span
                              className="text-xs font-mono"
                              style={{ color: theme.textSecondary }}
                            >
                              [{segment.start} - {segment.end}] {segment.speaker ? `• ${segment.speaker}` : ''}
                            </span>
                            <button
                              onClick={() => handleCopySegment(segment.text)}
                              className="opacity-0 group-hover:opacity-100 transition-opacity"
                              title="Copy segment"
                            >
                              <Copy className="w-3 h-3" style={{ color: theme.textSecondary }} />
                            </button>
                          </div>
                          <p
                            className="text-sm leading-relaxed"
                            style={{ color: theme.text }}
                            dir="rtl"
                            dangerouslySetInnerHTML={{ __html: 'highlightedHtml' in segment ? segment.highlightedHtml : segment.text }}
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Bottom Actions */}
              <div className="flex justify-between items-center mt-4">
                <p className="text-xs" style={{ color: theme.textSecondary }}>
                  {selectedFile?.status === 'completed' && `${currentTranscription.length} segments`}
                </p>
                <Button
                  onClick={handleClearAll}
                  variant="outline"
                  style={{ borderColor: theme.border, backgroundColor: theme.cardBg, color: theme.text }}
                >
                  Clear All
                </Button>
              </div>
            </div>
          </div>
        </div>
      </Resizable >
    </div >
  );
}
