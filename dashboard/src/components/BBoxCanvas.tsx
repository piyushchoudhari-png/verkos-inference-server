import { useEffect, useState } from 'react';
import type { Detection } from '../types';

interface Props {
  frameHandle: FileSystemFileHandle | undefined;
  detections: Detection[];
  resolutionW: number;
  resolutionH: number;
}

export function BBoxCanvas({ frameHandle, detections, resolutionW: _rw, resolutionH: _rh }: Props) {
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [imgError, setImgError] = useState(false);

  useEffect(() => {
    let url: string | null = null;
    if (frameHandle) {
      frameHandle.getFile().then(f => {
        url = URL.createObjectURL(f);
        setImgUrl(url);
        setImgError(false);
      }).catch(() => setImgError(true));
    } else {
      setImgUrl(null);
    }
    return () => { if (url) URL.revokeObjectURL(url); };
  }, [frameHandle]);

  if (imgError || (!frameHandle && detections.length === 0)) {
    return (
      <div className="w-full h-32 bg-gray-100 rounded flex items-center justify-center text-gray-400 text-xs">
        {imgError ? 'frame not saved' : 'no frame'}
      </div>
    );
  }

  return (
    <div className="relative w-full bg-black rounded overflow-hidden">
      {imgUrl ? (
        <img src={imgUrl} alt="frame" className="w-full block" />
      ) : !frameHandle ? (
        <div className="h-32 flex items-center justify-center text-gray-500 text-xs">frame not saved</div>
      ) : (
        <div className="h-32 flex items-center justify-center text-gray-400 text-xs animate-pulse">loading…</div>
      )}
    </div>
  );
}
