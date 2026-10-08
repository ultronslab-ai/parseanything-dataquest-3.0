/**
 * ParseAnything — Interactive Universal Document Ingestion Frontend
 * Provenance-aware Bounding Box Inspection, 3-Pane Dashboard, 1-Click Demonstrations
 */

// Application State
const state = {
  currentDocument: null,
  currentPage: 1,
  selectedBlockId: null,
  scaleX: 1.0,
  scaleY: 1.0,
  naturalWidth: 612.0,
  naturalHeight: 792.0,
  zoomLevel: 1.0
};

// DOM References
const elements = {
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  btnBrowse: document.getElementById('btnBrowse'),
  btnRunBenchmarks: document.getElementById('btnRunBenchmarks'),
  benchmarkModal: document.getElementById('benchmarkModal'),
  btnCloseModal: document.getElementById('btnCloseModal'),
  modalBenchmarkBody: document.getElementById('modalBenchmarkBody'),

  // Stepper
  stepUpload: document.getElementById('stepUpload'),
  stepDetect: document.getElementById('stepDetect'),
  stepExtract: document.getElementById('stepExtract'),
  stepAssemble: document.getElementById('stepAssemble'),
  stepVerify: document.getElementById('stepVerify'),

  // Header Telemetry
  headCostPill: document.getElementById('headCostPill'),
  headSpeedPill: document.getElementById('headSpeedPill'),

  // Banner
  bannerFilename: document.getElementById('bannerFilename'),
  bannerStatus: document.getElementById('bannerStatus'),
  bannerFormat: document.getElementById('bannerFormat'),
  bannerParser: document.getElementById('bannerParser'),
  bannerPageType: document.getElementById('bannerPageType'),
  bannerHandwriting: document.getElementById('bannerHandwriting'),
  bannerMeta: document.getElementById('bannerMeta'),
  bannerTablesCount: document.getElementById('bannerTablesCount'),
  bannerFiguresCount: document.getElementById('bannerFiguresCount'),
  bannerEquationsCount: document.getElementById('bannerEquationsCount'),
  bannerConfidenceAvg: document.getElementById('bannerConfidenceAvg'),
  bannerFormatIcon: document.getElementById('bannerFormatIcon'),

  // Left Pane
  previewViewport: document.getElementById('previewViewport'),
  nonVisualFallback: document.getElementById('nonVisualFallback'),
  fallbackRangePill: document.getElementById('fallbackRangePill'),
  btnZoomOut: document.getElementById('btnZoomOut'),
  btnZoomIn: document.getElementById('btnZoomIn'),
  zoomIndicator: document.getElementById('zoomIndicator'),

  // Center Pane
  blocksList: document.getElementById('blocksList'),
  blocksCountBadge: document.getElementById('blocksCountBadge'),

  // Right Pane (Inspector Tabs)
  tabButtons: document.querySelectorAll('.tab-btn'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  provMethodBadge: document.getElementById('provMethodBadge'),
  provConfBadge: document.getElementById('provConfBadge'),
  provOrderBadge: document.getElementById('provOrderBadge'),
  provBlockTitle: document.getElementById('provBlockTitle'),
  provSnippet: document.getElementById('provSnippet'),
  provDocVal: document.getElementById('provDocVal'),
  provPageVal: document.getElementById('provPageVal'),
  provBBoxVal: document.getElementById('provBBoxVal'),
  provLayerVal: document.getElementById('provLayerVal'),
  markdownRenderArea: document.getElementById('markdownRenderArea'),
  btnCopyMd: document.getElementById('btnCopyMd'),
  jsonCodeArea: document.getElementById('jsonCodeArea'),
  btnDownloadJson: document.getElementById('btnDownloadJson'),
  tablesInspectorList: document.getElementById('tablesInspectorList'),
  chartsViewContainer: document.getElementById('chartsViewContainer'),
  tabBtnEquations: document.getElementById('tabBtnEquations'),
  equationsCountHeader: document.getElementById('equationsCountHeader'),
  equationsInspectorList: document.getElementById('equationsInspectorList'),
  provBlockIdVal: document.getElementById('provBlockIdVal'),
  provModelVal: document.getElementById('provModelVal'),
  provReviewVal: document.getElementById('provReviewVal'),
  provEquationExtra: document.getElementById('provEquationExtra'),
  provEquationVal: document.getElementById('provEquationVal'),
  telCostVal: document.getElementById('telCostVal'),
  telSpeedVal: document.getElementById('telSpeedVal'),
  telPagesVal: document.getElementById('telPagesVal'),
  telBlocksVal: document.getElementById('telBlocksVal'),
  telLowConfVal: document.getElementById('telLowConfVal'),

  // Download Action Buttons
  btnDownloadBBox: document.getElementById('btnDownloadBBox'),
  btnDownloadMdMain: document.getElementById('btnDownloadMdMain'),
  btnDownloadJsonMain: document.getElementById('btnDownloadJsonMain'),
  btnDownloadAllMain: document.getElementById('btnDownloadAllMain'),
  btnDownloadMd: document.getElementById('btnDownloadMd'),
  btnDownloadProvenance: document.getElementById('btnDownloadProvenance'),
  tablesActionBar: document.getElementById('tablesActionBar'),
  tablesActionLabel: document.getElementById('tablesActionLabel'),
  btnDownloadAllTables: document.getElementById('btnDownloadAllTables'),
  chartsActionBar: document.getElementById('chartsActionBar'),
  chartsActionLabel: document.getElementById('chartsActionLabel'),
  btnDownloadAllCharts: document.getElementById('btnDownloadAllCharts'),

  // Previous Files list
  previousFilesList: document.getElementById('previousFilesList')
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  loadPreviousFiles();
});

let isProcessingFile = false;

function setupEventListeners() {
  // File Upload Handlers
  if (elements.fileInput) {
    // Crucial: stop propagation on fileInput click so it does NOT bubble up to dropzone
    // and trigger an infinite programmatic click recursion that browsers block!
    elements.fileInput.addEventListener('click', (e) => {
      e.stopPropagation();
    });

    elements.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });
  }

  if (elements.btnBrowse) {
    elements.btnBrowse.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (elements.fileInput) elements.fileInput.click();
    });
  }

  if (elements.dropzone) {
    elements.dropzone.addEventListener('click', (e) => {
      // Only invoke programmatic click if the user didn't already click directly on the fileInput
      if (e.target !== elements.fileInput && elements.fileInput) {
        elements.fileInput.click();
      }
    });

    elements.dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      elements.dropzone.classList.add('dragover');
    });

    elements.dropzone.addEventListener('dragleave', () => {
      elements.dropzone.classList.remove('dragover');
    });

    elements.dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      elements.dropzone.classList.remove('dragover');
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });
  }

  // 1-Click Sample Demo Buttons
  document.querySelectorAll('.demo-sample-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();
      const sample = btn.getAttribute('data-sample');
      if (sample) {
        await handleSampleIngest(sample);
      }
    });
  });

  // Download Handlers
  if (elements.btnDownloadMdMain) elements.btnDownloadMdMain.addEventListener('click', downloadCurrentMarkdown);
  if (elements.btnDownloadMd) elements.btnDownloadMd.addEventListener('click', downloadCurrentMarkdown);

  if (elements.btnDownloadJsonMain) elements.btnDownloadJsonMain.addEventListener('click', downloadCurrentJson);
  if (elements.btnDownloadJson) elements.btnDownloadJson.addEventListener('click', downloadCurrentJson);

  if (elements.btnDownloadProvenance) elements.btnDownloadProvenance.addEventListener('click', downloadCurrentProvenance);
  if (elements.btnDownloadAllMain) elements.btnDownloadAllMain.addEventListener('click', downloadAllCompleteResult);

  if (elements.btnDownloadAllTables) elements.btnDownloadAllTables.addEventListener('click', downloadAllTables);
  if (elements.btnDownloadAllCharts) elements.btnDownloadAllCharts.addEventListener('click', downloadAllFigures);

  if (elements.btnDownloadBBox) elements.btnDownloadBBox.addEventListener('click', downloadBBoxPdf);

  // Tab Switchers
  elements.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.tabButtons.forEach(b => b.classList.remove('active'));
      elements.tabPanes.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const targetId = 'tab' + btn.getAttribute('data-tab').charAt(0).toUpperCase() + btn.getAttribute('data-tab').slice(1);
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');
    });
  });

  // Zoom controls
  if (elements.btnZoomOut) elements.btnZoomOut.addEventListener('click', () => changeZoom(-0.1));
  if (elements.btnZoomIn) elements.btnZoomIn.addEventListener('click', () => changeZoom(0.1));

  // Markdown Copy Button
  elements.btnCopyMd.addEventListener('click', () => {
    if (state.currentDocument && state.currentDocument.markdown) {
      navigator.clipboard.writeText(state.currentDocument.markdown);
      elements.btnCopyMd.innerText = 'Copied!';
      setTimeout(() => elements.btnCopyMd.innerText = 'Copy Markdown', 2000);
    }
  });

  // Benchmarks Modal
  elements.btnRunBenchmarks.addEventListener('click', runBenchmarksModal);
  elements.btnCloseModal.addEventListener('click', () => elements.benchmarkModal.style.display = 'none');
  elements.benchmarkModal.addEventListener('click', (e) => {
    if (e.target === elements.benchmarkModal) elements.benchmarkModal.style.display = 'none';
  });

  // Recalibrate bounding boxes on viewport/window resize
  window.addEventListener('resize', () => {
    if (!state.currentDocument) return;
    const pages = state.currentDocument.pages || [{ page_number: 1, width: 612, height: 792 }];
    pages.forEach(pInfo => {
      recalibrateBoundingBoxes(pInfo.page_number, pInfo.width, pInfo.height);
    });
  });
}

// ============================================================================
// Pipeline Stepper Progress
// ============================================================================

function setPipelineStage(stage) {
  const steps = [elements.stepUpload, elements.stepDetect, elements.stepExtract, elements.stepAssemble, elements.stepVerify];
  steps.forEach(s => s.classList.remove('active', 'completed'));

  const indexMap = {
    'upload': 0,
    'detect': 1,
    'extract': 2,
    'assemble': 3,
    'verify': 4,
    'done': 5
  };

  const targetIdx = indexMap[stage] ?? 0;
  for (let i = 0; i < steps.length; i++) {
    if (i < targetIdx) steps[i].classList.add('completed');
    else if (i === targetIdx) steps[i].classList.add('active');
  }
}

// ============================================================================
// File Ingestion API Handlers
// ============================================================================

async function handleFileUpload(file) {
  if (!file || isProcessingFile) return;
  isProcessingFile = true;

  // Immediate UI feedback
  if (elements.bannerFilename) elements.bannerFilename.innerText = file.name;
  if (elements.bannerStatus) {
    elements.bannerStatus.innerText = 'PROCESSING...';
    elements.bannerStatus.className = 'badge badge-accent';
  }
  if (elements.bannerMeta) {
    elements.bannerMeta.innerText = `Ingesting ${file.name} (${(file.size / 1024).toFixed(1)} KB)...`;
  }
  setPipelineStage('upload');

  const formData = new FormData();
  formData.append('file', file);

  try {
    setPipelineStage('detect');
    setTimeout(() => setPipelineStage('extract'), 150);

    const res = await fetch('/api/v1/documents/upload', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      let errMsg = `Upload failed with HTTP ${res.status}`;
      try {
        const err = await res.json();
        if (err && err.detail) errMsg = err.detail;
      } catch (_) {
        const text = await res.text().catch(() => '');
        if (text) errMsg = `${errMsg}: ${text.slice(0, 100)}`;
      }
      throw new Error(errMsg);
    }

    setPipelineStage('assemble');
    const doc = await res.json();
    setPipelineStage('verify');
    setTimeout(() => setPipelineStage('done'), 200);

    if (doc.processing_status === 'failed') {
      const errMsg = (doc.errors && doc.errors.length > 0) ? doc.errors[0].message : 'Unsupported or corrupted document format.';
      alert(`Processing Failed: ${errMsg}`);
      if (elements.bannerStatus) {
        elements.bannerStatus.innerText = 'FAILED';
        elements.bannerStatus.className = 'badge badge-danger';
      }
      return;
    }

    renderDocument(doc);
    loadPreviousFiles();
  } catch (error) {
    console.error('Upload error:', error);
    alert(`Upload error: ${error.message}`);
    if (elements.bannerStatus) {
      elements.bannerStatus.innerText = 'ERROR';
      elements.bannerStatus.className = 'badge badge-danger';
    }
    setPipelineStage('upload');
  } finally {
    isProcessingFile = false;
    // Always clear input value so selecting the same file triggers 'change' every time!
    if (elements.fileInput) {
      elements.fileInput.value = '';
    }
  }
}

async function handleSampleIngest(filename) {
  if (isProcessingFile) return;
  isProcessingFile = true;

  if (elements.bannerFilename) elements.bannerFilename.innerText = filename;
  if (elements.bannerStatus) {
    elements.bannerStatus.innerText = 'INGESTING SAMPLE...';
    elements.bannerStatus.className = 'badge badge-accent';
  }
  if (elements.bannerMeta) {
    elements.bannerMeta.innerText = `Ingesting bundled sample ${filename}...`;
  }
  setPipelineStage('upload');

  try {
    setPipelineStage('detect');
    setTimeout(() => setPipelineStage('extract'), 150);

    const res = await fetch(`/api/v1/documents/sample/${encodeURIComponent(filename)}`, {
      method: 'POST'
    });

    if (!res.ok) {
      let errMsg = `Failed to process sample (HTTP ${res.status})`;
      try {
        const err = await res.json();
        if (err && err.detail) errMsg = err.detail;
      } catch (_) {
        const text = await res.text().catch(() => '');
        if (text) errMsg = `${errMsg}: ${text.slice(0, 100)}`;
      }
      throw new Error(errMsg);
    }

    setPipelineStage('assemble');
    const doc = await res.json();
    setPipelineStage('verify');
    setTimeout(() => setPipelineStage('done'), 200);

    if (doc.processing_status === 'failed') {
      const errMsg = (doc.errors && doc.errors.length > 0) ? doc.errors[0].message : 'Failed to ingest sample document.';
      alert(`Processing Failed: ${errMsg}`);
      if (elements.bannerStatus) {
        elements.bannerStatus.innerText = 'FAILED';
        elements.bannerStatus.className = 'badge badge-danger';
      }
      return;
    }

    renderDocument(doc);
    loadPreviousFiles();
  } catch (error) {
    console.error('Sample ingestion error:', error);
    alert(`Sample error: ${error.message}`);
    if (elements.bannerStatus) {
      elements.bannerStatus.innerText = 'ERROR';
      elements.bannerStatus.className = 'badge badge-danger';
    }
    setPipelineStage('upload');
  } finally {
    isProcessingFile = false;
  }
}

async function loadPreviousFiles() {
  try {
    const res = await fetch('/api/v1/documents');
    if (!res.ok) return;
    const files = await res.json();
    if (files.length === 0) return;

    elements.previousFilesList.innerHTML = '';
    files.forEach(f => {
      const btn = document.createElement('button');
      btn.className = 'demo-chip';
      btn.innerHTML = `
        <span class="chip-icon">📄</span>
        <div class="chip-meta">
          <span class="chip-name">${escapeHtml(f.filename)}</span>
          <span class="chip-desc">${f.type.toUpperCase()} • ${f.status}</span>
        </div>
      `;
      btn.addEventListener('click', () => loadDocumentById(f.id));
      elements.previousFilesList.appendChild(btn);
    });
  } catch (e) {
    console.error('Error fetching previous files:', e);
  }
}

async function loadDocumentById(docId) {
  setPipelineStage('detect');
  try {
    const res = await fetch(`/api/v1/documents/${docId}`);
    if (!res.ok) throw new Error('Document could not be loaded');
    const doc = await res.json();
    setPipelineStage('done');
    renderDocument(doc);
  } catch (error) {
    console.error('Error loading document:', error);
  }
}

async function downloadFromApi(url, defaultFilename) {
  try {
    const res = await fetch(url);
    if (!res.ok) {
      let errMsg = `Download failed (${res.status})`;
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) errMsg = errJson.detail;
      } catch (_) {}
      alert(errMsg);
      return;
    }
    const blob = await res.blob();
    const blobUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = blobUrl;

    const disposition = res.headers.get('Content-Disposition');
    let filename = defaultFilename;
    if (disposition && disposition.includes('filename=')) {
      const match = disposition.match(/filename=["']?([^"';]+)["']?/);
      if (match && match[1]) filename = match[1];
    }
    if (filename) a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(blobUrl);
  } catch (err) {
    alert(`Download error: ${err.message}`);
  }
}

function getDocStem() {
  if (!state.currentDocument) return 'document';
  return (state.currentDocument.filename || 'document').replace(/\.[^/.]+$/, '');
}

function downloadCurrentMarkdown() {
  if (!state.currentDocument) return;
  const docId = state.currentDocument.document_id;
  const stem = getDocStem();
  downloadFromApi(`/api/v1/documents/${docId}/download/markdown`, `${stem}.md`);
}

function downloadCurrentJson() {
  if (!state.currentDocument) return;
  const docId = state.currentDocument.document_id;
  const stem = getDocStem();
  downloadFromApi(`/api/v1/documents/${docId}/download/json`, `${stem}.json`);
}

function downloadCurrentProvenance() {
  if (!state.currentDocument) return;
  const docId = state.currentDocument.document_id;
  const stem = getDocStem();
  downloadFromApi(`/api/v1/documents/${docId}/download/provenance`, `${stem}_provenance.json`);
}

function downloadAllCompleteResult() {
  if (!state.currentDocument) return;
  const docId = state.currentDocument.document_id;
  const stem = getDocStem();
  downloadFromApi(`/api/v1/documents/${docId}/download/all`, `${stem}_complete_export.zip`);
}

function downloadAllTables() {
  if (!state.currentDocument) return;
  const docId = state.currentDocument.document_id;
  const stem = getDocStem();
  downloadFromApi(`/api/v1/documents/${docId}/download/tables`, `${stem}_tables.zip`);
}

function downloadAllFigures() {
  if (!state.currentDocument) return;
  const docId = state.currentDocument.document_id;
  const stem = getDocStem();
  downloadFromApi(`/api/v1/documents/${docId}/download/figures`, `${stem}_figures.zip`);
}

window.downloadSingleTableCsv = function(tableIdx) {
  if (!state.currentDocument) return;
  const docId = state.currentDocument.document_id;
  const stem = getDocStem();
  downloadFromApi(`/api/v1/documents/${docId}/download/table/${tableIdx}`, `${stem}_table_${tableIdx}.csv`);
};

window.downloadSingleFigureAsset = function(blockId) {
  if (!state.currentDocument) return;
  const docId = state.currentDocument.document_id;
  downloadFromApi(`/api/v1/documents/${docId}/download/figure/${blockId}`, `${blockId}.png`);
};

// ============================================================================
// Document Rendering Orchestration
// ============================================================================

function renderDocument(doc) {
  state.currentDocument = doc;
  state.currentPage = 1;
  state.selectedBlockId = null;
  state.zoomLevel = 1.0;
  if (elements.zoomIndicator) elements.zoomIndicator.innerText = '100%';
  applyZoom();

  // 1. Update Document Summary Banner
  elements.bannerFilename.innerText = doc.filename;
  elements.bannerStatus.innerText = doc.processing_status.toUpperCase();
  elements.bannerStatus.className = `badge ${doc.processing_status === 'success' ? 'badge-success' : 'badge-error'}`;
  elements.bannerFormat.innerText = (doc.file_type || 'PDF').toUpperCase();

  const method = doc.blocks.length > 0 ? doc.blocks[0].extraction_method : 'native';
  elements.bannerParser.innerText = method;

  const pageType = (doc.pages && doc.pages[0] && doc.pages[0].page_type) ? doc.pages[0].page_type : null;
  if (elements.bannerPageType) {
    if (pageType) {
      elements.bannerPageType.innerText = pageType;
      elements.bannerPageType.style.display = 'inline-block';
    } else {
      elements.bannerPageType.style.display = 'none';
    }
  }

  const hasHandwriting = doc.blocks.some(b => b.extraction_method === 'handwriting_ocr' || b.extraction_method?.includes('htr'));
  if (elements.bannerHandwriting) {
    elements.bannerHandwriting.style.display = hasHandwriting ? 'inline-block' : 'none';
  }

  const totalPages = doc.stats.pages_processed || 1;
  elements.bannerMeta.innerText = `${totalPages} Page${totalPages > 1 ? 's' : ''} • ${doc.stats.blocks_extracted} Blocks Extracted • Latency: ${doc.stats.processing_time_seconds}s`;

  elements.bannerTablesCount.innerText = doc.stats.tables_detected;
  elements.bannerFiguresCount.innerText = doc.stats.figures_detected;
  elements.bannerEquationsCount.innerText = doc.stats.equations_detected;

  const avgConf = doc.blocks.length > 0
    ? Math.round((doc.blocks.reduce((acc, b) => acc + b.confidence, 0) / doc.blocks.length) * 100)
    : 100;
  elements.bannerConfidenceAvg.innerText = `${avgConf}%`;

  // Update format icon
  const iconMap = { 'pdf': '📄', 'docx': '📝', 'pptx': '📊', 'xlsx': '📗', 'image': '🖼️' };
  elements.bannerFormatIcon.innerText = iconMap[doc.file_type] || '📄';

  // Header Telemetry Pill
  elements.headSpeedPill.innerText = `${doc.stats.pages_per_second} p/s`;
  elements.headCostPill.innerText = `$${doc.stats.estimated_cost_per_1000_pages_usd.toFixed(2)}`;

  // Enable download action buttons
  if (elements.btnDownloadBBox) elements.btnDownloadBBox.disabled = false;
  if (elements.btnDownloadMdMain) elements.btnDownloadMdMain.disabled = false;
  if (elements.btnDownloadJsonMain) elements.btnDownloadJsonMain.disabled = false;
  if (elements.btnDownloadAllMain) elements.btnDownloadAllMain.disabled = false;

  // 2. Render Left Pane (Document Page & Bounding Box Overlay)
  renderPagePreview();

  // 3. Render Center Pane (Semantic Blocks List)
  renderBlocksList();

  // 4. Render Right Pane Tabs
  renderMarkdownTab();
  renderJsonTab();
  renderTablesTab();
  renderChartsTab();
  renderEquationsTab();
  renderTelemetryTab();

  // Default select first block
  if (doc.blocks && doc.blocks.length > 0) {
    selectBlock(doc.blocks[0].block_id);
  }
}

// ============================================================================
// Left Pane: Document Preview & Bounding Boxes
// ============================================================================

function renderPagePreview() {
  const doc = state.currentDocument;

  // clean up existing dynamic containers
  const existingContainers = document.querySelectorAll('.dynamic-canvas-container');
  existingContainers.forEach(c => c.remove());

  if (doc.file_type === 'xlsx') {
    elements.nonVisualFallback.style.display = 'block';
    const firstTable = doc.blocks.find(b => b.type === 'table');
    elements.fallbackRangePill.innerText = firstTable && firstTable.source.cell_range
      ? `Sheet: ${firstTable.source.sheet} (${firstTable.source.cell_range})`
      : 'Spreadsheet Grid';
    return;
  }

  elements.nonVisualFallback.style.display = 'none';

  const pages = doc.pages && doc.pages.length > 0 ? doc.pages : [{
    page_number: 1,
    image_url: null,
    width: 612,
    height: 792
  }];

  pages.forEach(pInfo => {
    const container = document.createElement('div');
    container.className = 'preview-canvas-container dynamic-canvas-container';
    container.style.marginBottom = '20px';
    container.dataset.pageNum = pInfo.page_number;

    const img = document.createElement('img');
    img.className = 'page-image';
    img.id = `pagePreviewImg_${pInfo.page_number}`;

    const bboxLayer = document.createElement('div');
    bboxLayer.className = 'bbox-overlay-layer';
    bboxLayer.id = `bboxOverlayLayer_${pInfo.page_number}`;

    container.appendChild(img);
    container.appendChild(bboxLayer);
    elements.previewViewport.appendChild(container);

    if (pInfo.image_url) {
      img.src = pInfo.image_url;
      img.onload = () => recalibrateBoundingBoxes(pInfo.page_number, pInfo.width, pInfo.height);
    } else {
      createSyntheticPageCanvas(img, pInfo.width, pInfo.height, pInfo.page_number);
    }
  });
}

function createSyntheticPageCanvas(imgElem, w, h, pageNum) {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  ctx.strokeRect(0, 0, w, h);

  imgElem.src = canvas.toDataURL();
  imgElem.onload = () => recalibrateBoundingBoxes(pageNum, w, h);
}

function changeZoom(delta) {
  let newZoom = state.zoomLevel + delta;
  if (newZoom < 0.2) newZoom = 0.2;
  if (newZoom > 5.0) newZoom = 5.0;
  state.zoomLevel = newZoom;

  if (elements.zoomIndicator) {
    elements.zoomIndicator.innerText = `${Math.round(state.zoomLevel * 100)}%`;
  }

  applyZoom();
}

function applyZoom() {
  const canvasContainers = document.querySelectorAll('.dynamic-canvas-container');
  canvasContainers.forEach(container => {
    container.style.transform = `scale(${state.zoomLevel})`;
    container.style.transformOrigin = 'top left';
  });
}

function recalibrateBoundingBoxes(pageNum, natWidth, natHeight) {
  const doc = state.currentDocument;
  if (!doc) return;

  const imgElem = document.getElementById(`pagePreviewImg_${pageNum}`);
  if (!imgElem) return;

  const dispWidth = imgElem.clientWidth;
  const dispHeight = imgElem.clientHeight;
  if (dispWidth === 0 || dispHeight === 0) return;

  const scaleX = dispWidth / natWidth;
  const scaleY = dispHeight / natHeight;

  renderBoundingBoxes(pageNum, scaleX, scaleY);
}

function renderBoundingBoxes(pageNum, scaleX, scaleY) {
  const layer = document.getElementById(`bboxOverlayLayer_${pageNum}`);
  if (!layer) return;
  layer.innerHTML = '';

  const doc = state.currentDocument;
  if (!doc || !doc.blocks) return;

  const pageBlocks = doc.blocks.filter(b => {
    if (b.source.page !== undefined && b.source.page !== null) {
      return b.source.page === pageNum;
    }
    if (b.source.slide !== undefined && b.source.slide !== null) {
      return b.source.slide === pageNum;
    }
    return true; // Single sheet or image
  });

  pageBlocks.forEach(b => {
    if (!b.source.bbox) return;

    const [x0, y0, x1, y1] = b.source.bbox;
    const boxElem = document.createElement('div');

    const left = x0 * scaleX;
    const top = y0 * scaleY;
    const width = Math.max(12, (x1 - x0) * scaleX);
    const height = Math.max(12, (y1 - y0) * scaleY);

    boxElem.className = `bbox-rect ${getBBoxColorClass(b)}`;
    boxElem.id = `bbox_${b.block_id}`;
    boxElem.style.left = `${left}px`;
    boxElem.style.top = `${top}px`;
    boxElem.style.width = `${width}px`;
    boxElem.style.height = `${height}px`;
    boxElem.title = `[#${b.reading_order}] ${b.type.toUpperCase()} | Conf: ${Math.round(b.confidence * 100)}%`;

    boxElem.innerText = b.content || "";
    boxElem.style.overflow = "hidden";
    boxElem.style.textOverflow = "ellipsis";
    boxElem.style.whiteSpace = "nowrap";
    boxElem.style.fontSize = "11px";
    boxElem.style.padding = "2px";
    boxElem.style.fontWeight = "600";
    boxElem.style.color = "#0f172a";

    if (b.block_id === state.selectedBlockId) {
      boxElem.classList.add('highlighted');
    }

    boxElem.addEventListener('click', (e) => {
      e.stopPropagation();
      selectBlock(b.block_id, true);
    });

    layer.appendChild(boxElem);
  });
}

function getBBoxColorClass(block) {
  if (block.confidence < 0.70) return 'color-low';
  switch (block.type) {
    case 'table': return 'color-table';
    case 'chart':
    case 'figure': return 'color-chart';
    case 'equation': return 'color-math';
    default: 
      if (block.extraction_method === 'handwriting_ocr') return 'color-orange';
      return 'color-text';
  }
}

// ============================================================================
// Center Pane: Semantic Blocks Stream
// ============================================================================

function renderBlocksList() {
  elements.blocksList.innerHTML = '';
  const doc = state.currentDocument;
  if (!doc || !doc.blocks) return;

  elements.blocksCountBadge.innerText = `${doc.blocks.length} Blocks`;

  doc.blocks.forEach(block => {
    const card = document.createElement('div');
    card.className = `block-card ${block.block_id === state.selectedBlockId ? 'selected' : ''}`;
    card.id = `card_${block.block_id}`;

    // Header with badges
    const confBadgeClass = block.confidence >= 0.90 ? 'badge-success' : (block.confidence >= 0.70 ? 'badge-warn' : 'badge-error');

    let renderedContent = '';
    if (block.type === 'heading') {
      renderedContent = `<div class="block-content-view is-heading">${escapeHtml(block.content)}</div>`;
    } else if (block.type === 'table' && block.table_data && block.table_data.headers.length > 0) {
      renderedContent = renderHtmlMiniTable(block.table_data);
    } else if (block.type === 'equation') {
      const latexStr = block.equation_data ? block.equation_data.latex : block.content;
      renderedContent = `<div class="block-content-view is-equation">$$ ${escapeHtml(latexStr)} $$</div>`;
    } else if (block.type === 'chart' || block.type === 'figure') {
      const title = block.chart_data && block.chart_data.title ? block.chart_data.title : 'Figure/Chart';
      renderedContent = `<div class="block-content-view"><strong>📈 [${title}]</strong><br><span style="color:var(--text-muted);font-size:12px;">Visual figure detected & preserved with bounding box.</span></div>`;
    } else {
      renderedContent = `<div class="block-content-view">${escapeHtml(block.content)}</div>`;
    }

    // Add specific text for Handwriting
    let displayType = block.type.toUpperCase();
    if (block.extraction_method === 'handwriting_ocr') {
        displayType = 'HANDWRITING';
    } else if (block.extraction_method === 'math_ocr' || block.type === 'equation') {
        displayType = 'EQUATION';
    }

    let statusHtml = '';
    const isNeedsReview = block.status === 'needs_review' || block.status === 'review_required' || block.confidence < 0.70 || (block.metadata && block.metadata.requires_review) || block.requires_review;
    if (isNeedsReview) {
        statusHtml = `<span style="color:var(--accent-amber);font-size:12px;margin-left:8px;font-weight:bold;">⚠ Needs Review</span>`;
    }

    card.innerHTML = `
      <div class="block-card-header">
        <div class="block-badges">
          <span class="order-pill">#${block.reading_order}</span>
          <span class="badge badge-neutral">${displayType}</span>
          <span class="badge ${confBadgeClass}">${Math.round(block.confidence * 100)}%</span>
          ${statusHtml}
        </div>
        <span class="badge badge-method">${block.extraction_method}</span>
      </div>
      ${renderedContent}
      <div class="block-card-footer">
        <span>${formatBlockLocation(block.source)}</span>
        <button class="prov-btn" onclick="selectBlock('${block.block_id}', true)">
          Inspect Provenance →
        </button>
      </div>
    `;

    card.addEventListener('click', () => selectBlock(block.block_id, true));
    elements.blocksList.appendChild(card);
  });
}

function renderHtmlMiniTable(td) {
  let html = '<div style="overflow-x:auto;"><table class="mini-table">';
  if (td.headers.length > 0) {
    html += '<thead>';
    td.headers.forEach(hRow => {
      html += '<tr>' + hRow.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr>';
    });
    html += '</thead>';
  }
  html += '<tbody>';
  td.rows.slice(0, 5).forEach(row => {
    html += '<tr>' + row.map(c => `<td>${escapeHtml(c)}</td>`).join('') + '</tr>';
  });
  if (td.rows.length > 5) {
    html += `<tr><td colspan="${td.num_cols}" style="text-align:center;color:var(--text-dim);font-style:italic;">... +${td.rows.length - 5} more rows</td></tr>`;
  }
  html += '</tbody></table></div>';
  return html;
}

function formatBlockLocation(source) {
  const parts = [];
  if (source.page) parts.push(`Page ${source.page}`);
  if (source.slide) parts.push(`Slide ${source.slide}`);
  if (source.sheet) parts.push(`Sheet '${source.sheet}'`);
  if (source.cell_range) parts.push(source.cell_range);
  if (source.bbox) parts.push(`[${source.bbox.map(x => Math.round(x)).join(', ')}]`);
  return parts.join(' • ') || 'Source Root';
}

// ============================================================================
// Block Selection & Provenance Highlighting (Star Differentiator)
// ============================================================================

window.selectBlock = function (blockId, scrollIntoView = false) {
  state.selectedBlockId = blockId;
  const doc = state.currentDocument;
  if (!doc) return;

  const block = doc.blocks.find(b => b.block_id === blockId);
  if (!block) return;

  // Highlight center card
  document.querySelectorAll('.block-card').forEach(c => c.classList.remove('selected'));
  const cardElem = document.getElementById(`card_${blockId}`);
  if (cardElem) {
    cardElem.classList.add('selected');
    if (scrollIntoView) {
      cardElem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  // Highlight bounding box on preview
  document.querySelectorAll('.bbox-rect').forEach(r => r.classList.remove('highlighted'));
  const bboxElem = document.getElementById(`bbox_${blockId}`);
  if (bboxElem) {
    bboxElem.classList.add('highlighted');
  }

  // Update Provenance Tab in Right Pane
  elements.provMethodBadge.innerText = block.extraction_method;
  elements.provConfBadge.innerText = `${Math.round(block.confidence * 100)}% Confidence`;
  elements.provConfBadge.className = `badge ${block.confidence >= 0.90 ? 'badge-success' : (block.confidence >= 0.70 ? 'badge-warn' : 'badge-error')}`;
  elements.provOrderBadge.innerText = `Reading Order: #${block.reading_order}`;

  elements.provBlockTitle.innerText = `${block.type.toUpperCase()}: Block #${block.reading_order}`;
  elements.provSnippet.innerText = block.content.slice(0, 140) + (block.content.length > 140 ? '...' : '');

  elements.provDocVal.innerText = block.source.file || doc.filename;
  elements.provPageVal.innerText = block.source.page ? `Page ${block.source.page}` : (block.source.slide ? `Slide ${block.source.slide}` : (block.source.sheet ? `Sheet ${block.source.sheet}` : 'Root'));

  if (elements.provBlockIdVal) {
    elements.provBlockIdVal.innerText = `${block.block_id} (${block.type})`;
  }
  if (elements.provModelVal) {
    elements.provModelVal.innerText = block.model_name || (block.extraction_method === 'handwriting_ocr' ? 'HTR Engine' : (block.extraction_method === 'math_ocr' ? 'Math OCR' : block.extraction_method));
  }
  if (elements.provReviewVal) {
    const isNeedsReview = block.status === 'needs_review' || block.status === 'review_required' || block.confidence < 0.70 || (block.metadata && block.metadata.requires_review) || block.requires_review;
    elements.provReviewVal.innerHTML = isNeedsReview 
      ? '<span style="color:var(--accent-amber);font-weight:600;">⚠ Needs Review</span>' 
      : '<span style="color:var(--accent-emerald);font-weight:600;">✓ Verified</span>';
  }

  elements.provBBoxVal.innerText = block.source.bbox ? `[${block.source.bbox.map(x => Number(x).toFixed(1)).join(', ')}]` : (block.source.cell_range || 'N/A');
  elements.provLayerVal.innerText = block.extraction_method.includes('ocr') ? 'Level 2 (Specialized OCR Engine)' : 'Level 1 (Deterministic Native)';

  if (elements.provEquationExtra) {
    if (block.type === 'equation' || block.equation_data) {
      elements.provEquationExtra.style.display = 'block';
      const eqId = (block.equation_data && block.equation_data.equation_id) ? block.equation_data.equation_id : block.block_id;
      const latex = (block.equation_data && block.equation_data.latex) ? block.equation_data.latex : block.content;
      elements.provEquationVal.innerText = `ID: ${eqId}\nLaTeX: ${latex}`;
    } else {
      elements.provEquationExtra.style.display = 'none';
    }
  }
};

// ============================================================================
// Right Pane: Intelligence Tabs
// ============================================================================

function renderMarkdownTab() {
  const doc = state.currentDocument;
  if (!doc) return;
  elements.markdownRenderArea.innerText = doc.markdown || '';
}

function renderJsonTab() {
  const doc = state.currentDocument;
  if (!doc) return;
  elements.jsonCodeArea.innerText = JSON.stringify(doc, null, 2);
}

function renderTablesTab() {
  const doc = state.currentDocument;
  if (!doc) return;
  elements.tablesInspectorList.innerHTML = '';

  const tableBlocks = (doc.blocks || []).filter(b => b.type === 'table' && (b.table_data || (b.content && b.content.includes('|'))));
  const actionBar = elements.tablesActionBar || document.getElementById('tablesActionBar');
  if (actionBar) {
    if (tableBlocks.length > 0) {
      actionBar.style.display = 'flex';
      const countLabel = elements.tablesActionLabel || document.getElementById('tablesActionLabel');
      if (countLabel) countLabel.innerText = `Extracted Tables (${tableBlocks.length})`;
    } else {
      actionBar.style.display = 'none';
    }
  }

  if (tableBlocks.length === 0) {
    elements.tablesInspectorList.innerHTML = '<p style="color:var(--text-dim);font-style:italic;">No tabular structures detected in this document.</p>';
    return;
  }

  tableBlocks.forEach((tbl, idx) => {
    const wrap = document.createElement('div');
    wrap.style.marginBottom = '20px';
    wrap.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <h4 style="color:#fff;margin:0;">Table ${idx + 1} (${tbl.source.page ? `Page ${tbl.source.page}` : tbl.source.sheet || 'Grid'})</h4>
        <div style="display:flex;align-items:center;gap:8px;">
          <span class="badge badge-success">Merged Cells: ${tbl.table_data ? tbl.table_data.merged_cells_count : 0}</span>
          <button class="btn btn-outline btn-xs" style="border-color:var(--accent-emerald);color:var(--accent-emerald);" onclick="downloadSingleTableCsv(${idx + 1})">
            Download CSV
          </button>
        </div>
      </div>
      ${tbl.table_data ? renderHtmlMiniTable(tbl.table_data) : `<pre>${escapeHtml(tbl.content)}</pre>`}
    `;
    elements.tablesInspectorList.appendChild(wrap);
  });
}

function renderChartsTab() {
  const doc = state.currentDocument;
  if (!doc) return;
  const container = elements.chartsViewContainer || document.getElementById('chartsViewContainer');
  if (!container) return;

  const chartBlocks = (doc.blocks || []).filter(b => b.type === 'chart' || b.type === 'figure');
  const actionBar = elements.chartsActionBar || document.getElementById('chartsActionBar');
  if (actionBar) {
    if (chartBlocks.length > 0) {
      actionBar.style.display = 'flex';
      const countLabel = elements.chartsActionLabel || document.getElementById('chartsActionLabel');
      if (countLabel) countLabel.innerText = `Extracted Figures & Charts (${chartBlocks.length})`;
    } else {
      actionBar.style.display = 'none';
    }
  }

  if (chartBlocks.length === 0) {
    container.innerHTML = '<p style="color:var(--text-dim);font-style:italic;">No charts or figures detected in this document.</p>';
    return;
  }

  container.innerHTML = '';
  chartBlocks.forEach((ch, idx) => {
    const card = document.createElement('div');
    card.style.cssText = 'background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-bottom: 12px;';
    const title = (ch.chart_data && ch.chart_data.title) ? ch.chart_data.title : `Figure #${idx + 1}`;
    const pageNum = ch.source ? (ch.source.page || 1) : 1;
    const bboxStr = ch.source && ch.source.bbox ? `[${ch.source.bbox.map(x => Math.round(x)).join(', ')}]` : 'N/A';
    card.innerHTML = `
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <h4 style="color:#fff;margin:0;">📈 ${escapeHtml(title)}</h4>
        <span class="badge badge-neutral">Page ${pageNum}</span>
      </div>
      <div style="font-size:12px;color:var(--text-muted);margin-bottom:8px;">
        ${escapeHtml(ch.content || 'Visual structure preserved')}
      </div>
      <div style="font-size:11px;color:var(--text-dim);margin-bottom:8px;">
        Bounding Box: <code style="color:#cbd5e1;">${bboxStr}</code>
      </div>
      <div style="display:flex;gap:6px;align-items:center;">
        <button class="prov-btn" style="padding:4px 8px;font-size:11px;" onclick="selectBlock('${ch.block_id}', true); document.getElementById('tabBtnProv').click();">
          Inspect Provenance →
        </button>
        <button class="btn btn-outline btn-xs" style="border-color:var(--accent-purple);color:var(--accent-purple);padding:4px 8px;font-size:11px;" onclick="downloadSingleFigureAsset('${ch.block_id}')">
          Download Asset
        </button>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderEquationsTab() {
  const doc = state.currentDocument;
  if (!doc) return;
  const listElem = elements.equationsInspectorList || document.getElementById('equationsInspectorList');
  const countHeader = elements.equationsCountHeader || document.getElementById('equationsCountHeader');
  if (!listElem) return;

  const eqBlocks = (doc.blocks || []).filter(b => b.type === 'equation' || b.equation_data);
  const count = eqBlocks.length;
  if (countHeader) {
    countHeader.innerText = `EQUATIONS: ${count}`;
  }

  if (count === 0) {
    listElem.innerHTML = `
      <div style="color: var(--text-muted); font-size: 13px; padding: 16px 0; text-align: center;">
        Equations: 0<br><span style="font-size: 12px; color: var(--text-dim); margin-top: 4px; display: inline-block;">No mathematical equations detected in this document.</span>
      </div>
    `;
    return;
  }

  listElem.innerHTML = '';
  eqBlocks.forEach((eq, idx) => {
    const isNeedsReview = eq.status === 'needs_review' || eq.status === 'review_required' || eq.confidence < 0.70 || (eq.metadata && eq.metadata.requires_review) || eq.requires_review;
    const isHighConf = !isNeedsReview && eq.confidence >= 0.70;
    const confPct = Math.round(eq.confidence * 100);
    const latexText = eq.equation_data && eq.equation_data.latex ? eq.equation_data.latex : eq.content;
    const rawText = eq.equation_data && eq.equation_data.raw_text ? eq.equation_data.raw_text : eq.content;
    const pageNum = eq.source ? (eq.source.page || 1) : 1;
    const bboxStr = eq.source && eq.source.bbox ? `[${eq.source.bbox.map(x => Math.round(x)).join(', ')}]` : 'N/A';
    const methodStr = eq.extraction_method || 'math_ocr';

    const card = document.createElement('div');
    card.className = 'equation-card';
    card.style.cssText = 'background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 14px; margin-bottom: 14px; transition: all 0.2s ease;';

    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 6px;">
        <span style="font-weight: 700; color: #f8fafc; font-size: 14px;">Equation #${idx + 1}</span>
        <span class="badge ${isHighConf ? 'badge-success' : 'badge-warn'}" style="font-size: 12px; font-weight: 700;">${confPct}%</span>
      </div>

      <div style="font-size: 15px; color: #38bdf8; font-weight: 600; margin-bottom: 8px; word-break: break-all;">
        ${escapeHtml(rawText)}
      </div>

      <div style="margin-bottom: 10px;">
        <div style="font-size: 11px; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.05em; margin-bottom: 2px;">LaTeX:</div>
        <pre style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 4px; padding: 6px 10px; font-size: 12px; color: #a5f3fc; overflow-x: auto; margin: 0; font-family: monospace;">${escapeHtml(latexText)}</pre>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 11px; color: var(--text-muted); margin-bottom: 10px;">
        <div><strong>Page:</strong> ${pageNum}</div>
        <div><strong>Method:</strong> <span class="badge badge-method" style="font-size: 10px; padding: 2px 6px;">${escapeHtml(methodStr)}</span></div>
        <div style="grid-column: span 2;"><strong>Bounding Box:</strong> <code style="color: #cbd5e1; font-size: 11px;">${bboxStr}</code></div>
      </div>

      <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.06);">
        ${isHighConf 
          ? '<span style="color: #4ade80; font-size: 12px; font-weight: 600;">✓ High Confidence</span>' 
          : '<span style="color: #f59e0b; font-size: 12px; font-weight: 600;">⚠ Needs Review</span>'}
        <button class="prov-btn" style="padding: 4px 10px; font-size: 11px;" onclick="selectBlock('${eq.block_id}', true); document.getElementById('tabBtnProv').click();">
          Inspect Provenance →
        </button>
      </div>
    `;

    listElem.appendChild(card);
  });
}

function renderTelemetryTab() {
  const doc = state.currentDocument;
  if (!doc) return;

  elements.telCostVal.innerText = `$${doc.stats.estimated_cost_per_1000_pages_usd.toFixed(2)}`;
  elements.telSpeedVal.innerText = `${doc.stats.pages_per_second} p/s`;
  elements.telPagesVal.innerText = doc.stats.pages_processed;
  elements.telBlocksVal.innerText = doc.stats.blocks_extracted;
  elements.telLowConfVal.innerText = doc.stats.low_confidence_blocks;
}

// ============================================================================
// Benchmark Runner Modal Dialog
// ============================================================================

async function runBenchmarksModal() {
  elements.benchmarkModal.style.display = 'flex';
  elements.modalBenchmarkBody.innerHTML = '<div class="loader-spinner">Loading 13x automated benchmark suites status...</div>';

  try {
    const res = await fetch('/api/v1/benchmarks');
    if (!res.ok) throw new Error('Failed to load benchmark suites status');
    const data = await res.json();
    renderBenchmarkSuitesModal(data);
  } catch (error) {
    elements.modalBenchmarkBody.innerHTML = `<div style="color:var(--accent-rose);padding:16px;">Failed to load benchmark suites: ${escapeHtml(error.message)}</div>`;
  }
}

function renderBenchmarkSuitesModal(data) {
  const allExecuted = Boolean(data.has_official_results || (data.executed_suites_count === (data.total_suites_configured || 13) && data.executed_suites_count > 0));
  const allAvailable = Boolean(data.available_datasets_count === (data.total_suites_configured || 13));

  let bannerTitle = 'ℹ️ No official benchmark dataset or official benchmark results available';
  let bannerDesc = 'The 13 automated benchmark suites below are configured and ready to execute once official test documents are provided. Suites without test files are marked as <strong>Dataset not available</strong>, and unexecuted suites display <strong>No benchmark results available</strong> (never 0%, failed, or fabricated).';
  let bannerColor = 'rgba(148, 163, 184, 0.2)';

  if (allExecuted) {
    bannerTitle = '✓ 13x Automated Benchmark Test Suite Executed';
    bannerDesc = 'All 13 benchmark suites executed on genuine multi-format test documents. Real processing latencies, semantic blocks, and table detections are shown below.';
    bannerColor = 'rgba(16, 185, 129, 0.3)';
  } else if (allAvailable) {
    bannerTitle = '⚡ 13x Automated Benchmark Test Datasets Available';
    bannerDesc = 'All 13 benchmark test dataset files are ready in <code>test_documents/</code>. Click <strong>Execute Benchmark Runner</strong> to execute all test cases, or download individual dataset files below.';
    bannerColor = 'rgba(56, 189, 248, 0.3)';
  }

  let html = `
    <!-- Benchmark Notice & Actions Banner -->
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid ${bannerColor}; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 280px;">
          <div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
            ${bannerTitle}
          </div>
          <div style="font-size: 12px; color: #94a3b8; line-height: 1.5;">
            ${bannerDesc}
          </div>
        </div>
        <div style="display: flex; gap: 8px; flex-shrink: 0; flex-wrap: wrap; align-items: center;">
          <button class="btn btn-primary btn-xs" id="btnRunBenchmarkRunner" style="font-weight: 600;">
            ▶ Execute Benchmark Runner
          </button>
          <button class="btn btn-outline btn-xs" id="btnRunSmokeTest" style="border-color: var(--accent-amber); color: var(--accent-amber);">
            ⚡ Local Smoke Test
          </button>
          <a href="/api/v1/benchmarks/download/dataset" class="btn btn-outline btn-xs" download style="border-color: #38bdf8; color: #38bdf8; text-decoration: none;" title="Download all 13 test files as a ZIP">
            📦 Download All Datasets (.zip)
          </a>
          <a href="/api/v1/benchmarks/download/results" class="btn btn-outline btn-xs" download style="border-color: #a78bfa; color: #a78bfa; text-decoration: none;" title="Download benchmark results report as JSON">
            📊 Download Results (.json)
          </a>
        </div>
      </div>
    </div>

    <!-- Status Overview Pills -->
    <div style="display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; align-items: center;">
      <span class="badge" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-size: 12px; padding: 5px 12px;">
        Configured Suites: ${data.total_suites_configured || 13}
      </span>
      <span class="badge" style="background: ${allAvailable ? 'rgba(16, 185, 129, 0.15)' : 'rgba(148, 163, 184, 0.15)'}; color: ${allAvailable ? '#10b981' : '#94a3b8'}; font-size: 12px; padding: 5px 12px;">
        Datasets Available: ${data.available_datasets_count || 0} / ${data.total_suites_configured || 13}
      </span>
      <span class="badge" style="background: ${allExecuted ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)'}; color: ${allExecuted ? '#10b981' : '#f59e0b'}; font-size: 12px; padding: 5px 12px;">
        Execution Status: ${escapeHtml(data.overall_status_display || 'No benchmark results available')}
      </span>
    </div>

    <!-- Local Smoke Test Results Container (if present) -->
    <div id="smokeTestResultsArea" style="display: none; margin-bottom: 16px;"></div>

    <!-- 13 Suites Table -->
    <div style="overflow-x: auto; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px;">
      <table class="mini-table" style="font-size: 12px; margin: 0; width: 100%;">
        <thead>
          <tr style="background: rgba(15, 23, 42, 0.8);">
            <th style="width: 36px; text-align: center;">#</th>
            <th>Suite Name</th>
            <th style="width: 60px;">Format</th>
            <th>Target Capability</th>
            <th>Dataset File (Click to Download)</th>
            <th style="width: 130px;">Dataset Status</th>
            <th style="width: 240px;">Execution Status / Metrics</th>
          </tr>
        </thead>
        <tbody>
  `;

  (data.suites || []).forEach(sc => {
    const datasetBadge = sc.dataset_available
      ? '<span class="badge badge-success" style="font-size: 11px;">Available</span>'
      : '<span class="badge" style="background: rgba(148, 163, 184, 0.15); color: #94a3b8; font-size: 11px;">Dataset not available</span>';

    let fileCellHtml = '';
    if (sc.dataset_available) {
      fileCellHtml = `
        <a href="/api/v1/benchmarks/download/file/${encodeURIComponent(sc.dataset_file)}" download class="dataset-download-link" title="Click to download ${escapeHtml(sc.dataset_file)}" style="display: inline-flex; align-items: center; gap: 5px; color: #38bdf8; text-decoration: none; font-weight: 500;">
          <span style="font-size: 11px;">⬇️</span>
          <code style="font-size: 11px; color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 2px 6px; border-radius: 4px;">${escapeHtml(sc.dataset_file)}</code>
        </a>
      `;
    } else {
      fileCellHtml = `<code style="font-size: 11px; color: #94a3b8;">${escapeHtml(sc.dataset_file)}</code>`;
    }

    let execStatusHtml = '';
    if (sc.execution_status === 'Dataset not available') {
      execStatusHtml = '<span style="color: #94a3b8; font-style: italic;">No benchmark results available</span>';
    } else if (sc.execution_status === 'Executed') {
      const blkInfo = sc.blocks_extracted !== undefined && sc.blocks_extracted !== null ? `${sc.blocks_extracted} blks` : '';
      const tblInfo = sc.tables_detected ? `, ${sc.tables_detected} tbl` : '';
      const metrics = [blkInfo + tblInfo, sc.latency_seconds ? `${sc.latency_seconds}s` : ''].filter(Boolean).join(' | ');
      execStatusHtml = `
        <span class="badge badge-success" style="font-size: 11px; display: inline-flex; align-items: center; gap: 4px;">
          ✓ ${escapeHtml(sc.result_display)}${metrics ? ` (${metrics})` : ''}
        </span>
      `;
    } else if (sc.execution_status === 'Ready to execute') {
      execStatusHtml = '<span class="badge" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-size: 11px;">Ready to execute</span>';
    } else {
      execStatusHtml = '<span style="color: #94a3b8; font-style: italic;">No benchmark results available</span>';
    }

    html += `
      <tr>
        <td style="text-align: center; color: var(--text-dim); font-weight: 600;">${sc.suite_number}</td>
        <td><strong>${escapeHtml(sc.name)}</strong></td>
        <td><span class="badge badge-neutral" style="font-size: 10px;">${escapeHtml(sc.format)}</span></td>
        <td style="color: var(--text-muted);">${escapeHtml(sc.capability)}</td>
        <td>${fileCellHtml}</td>
        <td>${datasetBadge}</td>
        <td>${execStatusHtml}</td>
      </tr>
    `;
  });

  html += `
        </tbody>
      </table>
    </div>
  `;

  elements.modalBenchmarkBody.innerHTML = html;

  // Wire runner buttons
  const btnRunner = document.getElementById('btnRunBenchmarkRunner');
  if (btnRunner) {
    btnRunner.addEventListener('click', executeBenchmarkRun);
  }

  const btnSmoke = document.getElementById('btnRunSmokeTest');
  if (btnSmoke) {
    btnSmoke.addEventListener('click', executeLocalSmokeTest);
  }
}

async function executeBenchmarkRun() {
  const btn = document.getElementById('btnRunBenchmarkRunner');
  if (btn) {
    btn.disabled = true;
    btn.innerText = 'Executing 13 Suites... Please wait';
  }

  try {
    const res = await fetch('/api/v1/benchmarks/run', { method: 'POST' });
    if (!res.ok) throw new Error('Benchmark runner request failed');
    const data = await res.json();
    renderBenchmarkSuitesModal(data);
  } catch (err) {
    alert(`Benchmark runner error: ${err.message}`);
    if (btn) {
      btn.disabled = false;
      btn.innerText = '▶ Execute Benchmark Runner';
    }
  }
}

async function executeLocalSmokeTest() {
  const btn = document.getElementById('btnRunSmokeTest');
  const smokeArea = document.getElementById('smokeTestResultsArea');
  if (btn) {
    btn.disabled = true;
    btn.innerText = 'Running Smoke Test...';
  }
  if (smokeArea) {
    smokeArea.style.display = 'block';
    smokeArea.innerHTML = '<div class="loader-spinner" style="font-size: 12px; padding: 12px;">Running local smoke test on bundled documents...</div>';
  }

  try {
    const res = await fetch('/api/v1/benchmarks/local-smoke-test', { method: 'POST' });
    if (!res.ok) throw new Error('Smoke test execution failed');
    const data = await res.json();

    let smokeHtml = `
      <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px;">
          <div>
            <span class="badge" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b; font-weight: 700; font-size: 12px; padding: 4px 10px;">
              ${escapeHtml(data.label)}
            </span>
            <span style="font-size: 12px; color: var(--text-dim); margin-left: 8px;">(Integrity check on ${data.tests_run} local files)</span>
          </div>
          <button class="btn btn-outline btn-xs" onclick="document.getElementById('smokeTestResultsArea').style.display='none';">Dismiss</button>
        </div>
        <div style="font-size: 12px; color: #cbd5e1; margin-bottom: 10px;">
          ${escapeHtml(data.disclaimer)}
        </div>
        <table class="mini-table" style="font-size: 12px; margin: 0;">
          <thead>
            <tr>
              <th>Test Target</th>
              <th>Source File</th>
              <th>Status</th>
              <th>Latency</th>
              <th>Blocks</th>
              <th>Avg Confidence</th>
              <th>Benchmark Classification</th>
            </tr>
          </thead>
          <tbody>
    `;

    (data.results || []).forEach(r => {
      smokeHtml += `
        <tr>
          <td><strong>${escapeHtml(r.test_name)}</strong></td>
          <td><code>${escapeHtml(r.file)}</code></td>
          <td><span class="badge ${r.status === 'PASSED' ? 'badge-success' : 'badge-error'}">${escapeHtml(r.status)}</span></td>
          <td>${r.latency_seconds}s</td>
          <td>${r.blocks_extracted}</td>
          <td>${r.average_confidence}%</td>
          <td><span style="color: #f59e0b; font-style: italic;">Local test — not an official benchmark.</span></td>
        </tr>
      `;
    });

    smokeHtml += '</tbody></table></div>';
    if (smokeArea) smokeArea.innerHTML = smokeHtml;
  } catch (err) {
    if (smokeArea) {
      smokeArea.innerHTML = `<div style="color: var(--accent-rose); font-size: 12px; padding: 8px;">Smoke test error: ${escapeHtml(err.message)}</div>`;
    }
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = '⚡ Run Local Smoke Test';
    }
  }
}

// Utility: HTML escaping
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

async function downloadBBoxPdf() {
  if (!state.currentDocument) return;
  const docId = state.currentDocument.document_id;
  elements.btnDownloadBBox.disabled = true;
  elements.btnDownloadBBox.innerHTML = '<span class="loader"></span> Preparing...';
  try {
    const res = await fetch(`/api/v1/documents/${docId}/export-pdf`);
    if (!res.ok) throw new Error('Failed to generate PDF');
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = `${state.currentDocument.filename.replace('.pdf', '')}_bboxes.pdf`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    alert(err.message);
  } finally {
    elements.btnDownloadBBox.disabled = false;
    elements.btnDownloadBBox.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg> Download PDF';
  }
}
