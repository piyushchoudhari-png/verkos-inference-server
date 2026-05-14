import { useMemo, useState } from 'react';
import type { MetricSample } from '../types';
import { parseDetections } from '../lib/parseDetections';
import { FrameCard } from './FrameCard';

interface Props {
  outputs: MetricSample[];
  frameHandles: Map<string, FileSystemFileHandle>;
}

type SortKey = 'index' | 'latency' | 'time';

function frameKey(s: MetricSample): string {
  return `feed${String(s.feed_id).padStart(2, '0')}_frame${String(s.frame_index).padStart(4, '0')}`;
}

export function FrameExplorer({ outputs, frameHandles }: Props) {
  const [filterFeed, setFilterFeed] = useState<number | 'all'>('all');
  const [filterStatus, setFilterStatus] = useState<'all' | 'ok' | 'error'>('all');
  const [filterEvent, setFilterEvent] = useState<string>('all');
  const [sortKey, setSortKey] = useState<SortKey>('index');
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 50;

  const feedIds = useMemo(() => [...new Set(outputs.map(s => s.feed_id))].sort((a, b) => a - b), [outputs]);
  const allEvents = useMemo(() => {
    const events = new Set<string>();
    for (const s of outputs) {
      const p = parseDetections(s.output_text);
      for (const d of p.detections) events.add(d.event);
    }
    return [...events].sort();
  }, [outputs]);

  const filtered = useMemo(() => {
    let result = outputs.slice();
    if (filterFeed !== 'all') result = result.filter(s => s.feed_id === filterFeed);
    if (filterStatus !== 'all') result = result.filter(s => s.status === filterStatus);
    if (filterEvent !== 'all') {
      result = result.filter(s => {
        const p = parseDetections(s.output_text);
        return p.detections.some(d => d.event === filterEvent);
      });
    }
    result.sort((a, b) => {
      if (sortKey === 'latency') return b.latency_s - a.latency_s;
      if (sortKey === 'time') return a.t0_epoch - b.t0_epoch;
      // index: sort by feed then frame
      if (a.feed_id !== b.feed_id) return a.feed_id - b.feed_id;
      return a.frame_index - b.frame_index;
    });
    return result;
  }, [outputs, filterFeed, filterStatus, filterEvent, sortKey]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const page_items = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  function resetPage() { setPage(0); }

  return (
    <section className="mb-6">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h2 className="text-lg font-semibold text-gray-800">Frames ({filtered.length})</h2>
        <div className="flex gap-2 flex-wrap">
          <select
            value={filterFeed}
            onChange={e => { setFilterFeed(e.target.value === 'all' ? 'all' : Number(e.target.value)); resetPage(); }}
            className="text-sm border border-gray-300 rounded px-2 py-1 bg-white"
          >
            <option value="all">All feeds</option>
            {feedIds.map(id => <option key={id} value={id}>Feed {id}</option>)}
          </select>
          <select
            value={filterStatus}
            onChange={e => { setFilterStatus(e.target.value as 'all' | 'ok' | 'error'); resetPage(); }}
            className="text-sm border border-gray-300 rounded px-2 py-1 bg-white"
          >
            <option value="all">All statuses</option>
            <option value="ok">OK</option>
            <option value="error">Error</option>
          </select>
          {allEvents.length > 0 && (
            <select
              value={filterEvent}
              onChange={e => { setFilterEvent(e.target.value); resetPage(); }}
              className="text-sm border border-gray-300 rounded px-2 py-1 bg-white"
            >
              <option value="all">All events</option>
              {allEvents.map(ev => <option key={ev} value={ev}>{ev}</option>)}
            </select>
          )}
          <select
            value={sortKey}
            onChange={e => setSortKey(e.target.value as SortKey)}
            className="text-sm border border-gray-300 rounded px-2 py-1 bg-white"
          >
            <option value="index">Sort: index</option>
            <option value="latency">Sort: latency ↓</option>
            <option value="time">Sort: time</option>
          </select>
        </div>
      </div>

      {filtered.length === 0 ? (
        <p className="text-gray-400 text-sm py-8 text-center">No frames match the current filters.</p>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {page_items.map(s => (
              <FrameCard
                key={frameKey(s)}
                sample={s}
                frameHandle={frameHandles.get(frameKey(s))}
              />
            ))}
          </div>
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2 mt-4">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="px-3 py-1 text-sm border rounded hover:bg-gray-50 disabled:opacity-40"
              >
                Prev
              </button>
              <span className="text-sm text-gray-600">Page {page + 1} / {totalPages}</span>
              <button
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                disabled={page === totalPages - 1}
                className="px-3 py-1 text-sm border rounded hover:bg-gray-50 disabled:opacity-40"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </section>
  );
}
