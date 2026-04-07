import { useEffect, useRef, useState } from 'react';
import styles from './FilterBar.module.css';

export interface FilterOption {
  value: string;
  label: string;
}

export interface FilterField {
  type: 'text' | 'select' | 'date';
  key: string;
  placeholder?: string;
  options?: FilterOption[];
}

interface FilterBarProps {
  fields: FilterField[];
  values: Record<string, string>;
  onChange: (key: string, value: string) => void;
  onClear: () => void;
  resultCount: number;
  totalCount: number;
}

export function FilterBar({ fields, values, onChange, onClear, resultCount, totalCount }: FilterBarProps) {
  const hasActive = Object.values(values).some(v => v !== '');

  return (
    <div className={styles.bar}>
      {fields.map(field => (
        field.type === 'text'
          ? <DebouncedTextInput
              key={field.key}
              value={values[field.key] ?? ''}
              placeholder={field.placeholder}
              onChange={v => onChange(field.key, v)}
            />
          : field.type === 'select'
          ? <select
              key={field.key}
              className={styles.select}
              value={values[field.key] ?? ''}
              onChange={e => onChange(field.key, e.target.value)}
            >
              {(field.options ?? []).map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          : <input
              key={field.key}
              type="date"
              className={styles.input}
              value={values[field.key] ?? ''}
              onChange={e => onChange(field.key, e.target.value)}
            />
      ))}
      {hasActive && (
        <button className={styles.clearBtn} onClick={onClear}>✕ 清除筛选</button>
      )}
      <span className={styles.count}>
        {hasActive ? `显示 ${resultCount} / ${totalCount} 条` : `共 ${totalCount} 条`}
      </span>
    </div>
  );
}

function DebouncedTextInput({
  value, placeholder, onChange,
}: { value: string; placeholder?: string; onChange: (v: string) => void }) {
  const [local, setLocal] = useState(value);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cancel pending timer on unmount
  useEffect(() => {
    return () => { if (timer.current) clearTimeout(timer.current); };
  }, []);

  // Sync if parent resets to empty (clear button)
  useEffect(() => {
    if (value === '') setLocal('');
  }, [value]);

  function handleChange(v: string) {
    setLocal(v);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => onChange(v), 300);
  }

  return (
    <input
      type="text"
      className={styles.input}
      placeholder={placeholder}
      value={local}
      onChange={e => handleChange(e.target.value)}
    />
  );
}
