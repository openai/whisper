interface ProgressProps {
  value: number;
  className?: string;
}

export default function Progress({ value, className }: ProgressProps) {
  return (
    <div className={`w-full bg-gray-200 rounded-full h-1.5 overflow-hidden ${className || ''}`}>
      <div
        className="bg-green-500 h-full transition-all duration-300"
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  );
}
