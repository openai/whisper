import { useState, useRef } from 'react';
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
  id: string;
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
}

// Get API URL from environment variable
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export default function App() {
  const [fileQueue, setFileQueue] = useState<FileItem[]>([]);
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const [isDark, setIsDark] = useState(false);
  const [windowSize, setWindowSize] = useState({ width: 1100, height: 700 });
  const [searchQuery, setSearchQuery] = useState('');
  const [exportFormat, setExportFormat] = useState('txt');
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

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files;
    if (!files) return;

    const newFiles: FileItem[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const newFileItem: FileItem = {
        id: `${Date.now()}-${i}`,
        name: file.name,
        status: 'pending',
        file: file,
      };
      newFiles.push(newFileItem);
    }

    setFileQueue([...fileQueue, ...newFiles]);
    if (!selectedFileId && newFiles.length > 0) {
      setSelectedFileId(newFiles[0].id);
    }
    toast.success(`${newFiles.length} file(s) added to queue`);

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleRemoveFile = (id: string) => {
    setFileQueue(fileQueue.filter(f => f.id !== id));
    if (selectedFileId === id) {
      setSelectedFileId(fileQueue[0]?.id || null);
    }
    toast.info('File removed from queue');
  };

  const handleTranscribe = async () => {
    if (!selectedFileId) return;

    const fileIndex = fileQueue.findIndex(f => f.id === selectedFileId);
    if (fileIndex === -1 || !fileQueue[fileIndex].file) return;

    // Update status to processing
    const updatedQueue = [...fileQueue];
    updatedQueue[fileIndex].status = 'processing';
    updatedQueue[fileIndex].progress = 0;
    setFileQueue(updatedQueue);

    try {
      const file = fileQueue[fileIndex].file!;
      const formData = new FormData();
      formData.append('file', file);
      formData.append('language', 'fa'); // Farsi by default

      // Show loading toast
      const loadingToastId = toast.loading('Loading Whisper model (first time only)...');

      const response = await fetch(`${API_URL}/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const result = await response.json();

      // Dismiss loading toast
      toast.dismiss(loadingToastId);

      if (result.status === 'success') {
        const updated = [...fileQueue];
        updated[fileIndex].status = 'completed';
        updated[fileIndex].progress = 100;
        updated[fileIndex].transcription = result.segments;
        updated[fileIndex].fullText = result.text;
        setFileQueue(updated);
        toast.success('Transcription completed!');
      } else {
        throw new Error(result.error || 'Unknown error');
      }
    } catch (error) {
      const updated = [...fileQueue];
      updated[fileIndex].status = 'error';
      setFileQueue(updated);
      const errorMsg = error instanceof Error ? error.message : 'Failed to transcribe file';
      toast.error(errorMsg);
    }
  };

  const handleCopySegment = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard');
  };

  const handleExport = async () => {
    const selectedFile = fileQueue.find(f => f.id === selectedFileId);
    if (!selectedFile?.transcription || !selectedFile.fullText) {
      toast.error('No transcription to export');
      return;
    }

    try {
      const toastId = toast.loading('Preparing export...');

      const response = await fetch(`${API_URL}/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          transcription: selectedFile.fullText,
          segments: selectedFile.transcription,
          format: exportFormat,
        }),
      });

      if (!response.ok) {
        throw new Error(`Export failed: ${response.statusText}`);
      }

      const result = await response.json();
      toast.dismiss(toastId);

      if (result.status === 'success') {
        // Create a blob and download
        const blob = new Blob([result.content], { type: result.mime_type });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${selectedFile.name.split('.')[0]}.${exportFormat === 'json' ? 'json' : exportFormat}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        toast.success(`Exported as ${exportFormat.toUpperCase()}`);
      } else {
        throw new Error(result.error || 'Export failed');
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Failed to export';
      toast.error(errorMsg);
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

  // Filter transcription based on search
  const filteredTranscription = searchQuery
    ? currentTranscription.filter(seg =>
        seg.text.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : currentTranscription;

  // Function to highlight search text
  const highlightText = (text: string, query: string) => {
    if (!query) return text;

    const parts = text.split(new RegExp(`(${query})`, 'gi'));
    return parts.map((part) =>
      part.toLowerCase() === query.toLowerCase()
        ? `<mark style="background-color: ${isDark ? '#4CAF50' : '#FFEB3B'}; color: ${isDark ? '#000' : '#000'}; padding: 2px 4px; border-radius: 2px;">${part}</mark>`
        : part
    ).join('');
  };

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

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="audio/*,video/*"
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
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
            className="w-64 border-r flex flex-col overflow-hidden"
            style={{ borderColor: theme.border, backgroundColor: theme.sidebarBg }}
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
                    onChange={(e) => setExportFormat(e.target.value as 'txt' | 'srt' | 'vtt' | 'json')}
                  >
                    <option value="txt">TXT</option>
                    <option value="srt">SRT</option>
                    <option value="vtt">VTT</option>
                    <option value="json">JSON</option>
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
                              [{segment.start} - {segment.end}]
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
                            dangerouslySetInnerHTML={{ __html: highlightText(segment.text, searchQuery) }}
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
      </Resizable>
    </div>
  );
}
