import html2pdf from 'html2pdf.js';

interface Props {
  runId: string;
}

export function ExportButton({ runId }: Props) {
  function handleExport() {
    const el = document.getElementById('dashboard-root');
    html2pdf()
      .set({
        margin: 10,
        filename: `verkos_run_${runId}.pdf`,
        image: { type: 'jpeg', quality: 0.92 },
        html2canvas: { scale: 2, useCORS: true },
        jsPDF: { unit: 'mm', format: 'a4', orientation: 'landscape' },
        pagebreak: { mode: ['avoid-all', 'css'] },
      })
      .from(el)
      .save();
  }

  return (
    <button
      onClick={handleExport}
      className="flex items-center gap-2 py-2 px-4 bg-gray-800 text-white rounded-lg text-sm font-medium hover:bg-gray-700"
    >
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      Export PDF
    </button>
  );
}
