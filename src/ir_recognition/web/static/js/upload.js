/**
 * IR Signature Recognition System - Upload Module
 *
 * Handles drag-and-drop image upload, file selection dialog, client-side
 * file type validation, image preview display, and recognition submission.
 *
 * Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
 */

import { apiRequest, showLoading, showError, isLocked, lock, unlock, setStatus, clearResults } from './app.js';
import { renderResults } from './results.js';
import { highlightStep, resetPipeline, showProcessingTime } from './pipeline.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/tiff'];
const ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

let dropzone = null;
let fileInput = null;
let imagePreview = null;
let fileInfo = null;

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/**
 * Initialize the upload module: get DOM references and wire event listeners.
 */
export function initUpload() {
    dropzone = document.getElementById('upload-dropzone');
    fileInput = document.getElementById('file-input');
    imagePreview = document.getElementById('image-preview');
    fileInfo = document.getElementById('file-info');

    if (!dropzone || !fileInput) return;

    // Drag-and-drop events
    dropzone.addEventListener('dragenter', handleDragEnter);
    dropzone.addEventListener('dragover', handleDragOver);
    dropzone.addEventListener('dragleave', handleDragLeave);
    dropzone.addEventListener('drop', handleDrop);

    // Click to open file dialog
    dropzone.addEventListener('click', handleDropzoneClick);

    // Keyboard activation (Enter/Space) for accessibility
    dropzone.addEventListener('keydown', handleDropzoneKeydown);

    // File input change
    fileInput.addEventListener('change', handleFileInputChange);
}

// ---------------------------------------------------------------------------
// Drag-and-Drop Handlers
// ---------------------------------------------------------------------------

/**
 * Handle dragenter: add visual feedback class.
 * @param {DragEvent} event
 */
function handleDragEnter(event) {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.add('dragover');
}

/**
 * Handle dragover: prevent default to allow drop.
 * @param {DragEvent} event
 */
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.add('dragover');
}

/**
 * Handle dragleave: remove visual feedback class.
 * @param {DragEvent} event
 */
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.remove('dragover');
}

/**
 * Handle drop: extract file and process it.
 * @param {DragEvent} event
 */
function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.remove('dragover');

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// ---------------------------------------------------------------------------
// Click / Keyboard Handlers
// ---------------------------------------------------------------------------

/**
 * Handle click on dropzone: trigger file input dialog.
 * @param {MouseEvent} event
 */
function handleDropzoneClick(event) {
    // Don't trigger if clicking on the file input itself
    if (event.target === fileInput) return;
    fileInput.click();
}

/**
 * Handle keyboard activation on dropzone (Enter or Space).
 * @param {KeyboardEvent} event
 */
function handleDropzoneKeydown(event) {
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        fileInput.click();
    }
}

/**
 * Handle file input change: get selected file and process it.
 * @param {Event} event
 */
function handleFileInputChange(event) {
    const files = event.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
    // Reset input so the same file can be re-selected
    fileInput.value = '';
}

// ---------------------------------------------------------------------------
// File Processing
// ---------------------------------------------------------------------------

/**
 * Validate file type, show preview and file info, then submit for recognition.
 * @param {File} file - The selected/dropped file
 */
function processFile(file) {
    if (!validateFileType(file)) {
        showError('Unsupported file format. Accepted: PNG, JPEG, TIFF');
        return;
    }

    showPreview(file);
    showFileInfo(file);
    submitForRecognition(file);
}

/**
 * Validate that the file is an accepted image type.
 * Checks MIME type first, then falls back to extension check
 * (some browsers don't set MIME type for TIFF).
 * @param {File} file
 * @returns {boolean} True if the file type is valid
 */
function validateFileType(file) {
    // Check MIME type
    if (file.type && ALLOWED_TYPES.includes(file.type)) {
        return true;
    }

    // Fallback: check file extension (handles TIFF on some browsers)
    const fileName = file.name.toLowerCase();
    return ALLOWED_EXTENSIONS.some(ext => fileName.endsWith(ext));
}

// ---------------------------------------------------------------------------
// Preview Display
// ---------------------------------------------------------------------------

/**
 * Display an image preview using FileReader.
 * @param {File} file
 */
function showPreview(file) {
    if (!imagePreview) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        imagePreview.innerHTML = '';
        const img = document.createElement('img');
        img.src = event.target.result;
        img.alt = `Preview of ${file.name}`;
        img.className = 'preview-image';
        imagePreview.appendChild(img);
    };
    reader.readAsDataURL(file);
}

/**
 * Display filename and formatted file size.
 * @param {File} file
 */
function showFileInfo(file) {
    if (!fileInfo) return;

    const sizeStr = formatFileSize(file.size);
    fileInfo.innerHTML = '';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'file-name';
    nameSpan.textContent = file.name;

    const sizeSpan = document.createElement('span');
    sizeSpan.className = 'file-size';
    sizeSpan.textContent = sizeStr;

    fileInfo.appendChild(nameSpan);
    fileInfo.appendChild(sizeSpan);
}

/**
 * Convert bytes to a human-readable file size string.
 * @param {number} bytes
 * @returns {string} Formatted size (e.g., "1.5 MB")
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';

    const units = ['B', 'KB', 'MB', 'GB'];
    const k = 1024;
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    const index = Math.min(i, units.length - 1);
    const size = bytes / Math.pow(k, index);

    // Use up to 1 decimal place, but drop trailing zero
    return `${size % 1 === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[index]}`;
}

// ---------------------------------------------------------------------------
// Recognition Submission
// ---------------------------------------------------------------------------

/**
 * Submit a file for recognition via the API.
 * Locks the UI, shows loading, posts the file, and renders results.
 * Exported so the generator module can also trigger recognition.
 *
 * @param {File} file - The image file to recognize
 */
export async function submitForRecognition(file) {
    if (isLocked()) return;

    lock();
    showLoading(true);
    clearResults();
    setStatus('Processing...', 'ready');

    // Highlight pipeline steps if available
    if (typeof resetPipeline === 'function') {
        resetPipeline();
    }
    if (typeof highlightStep === 'function') {
        highlightStep('input');
    }

    const formData = new FormData();
    formData.append('file', file);

    const startTime = performance.now();

    try {
        // Highlight preprocessing step
        if (typeof highlightStep === 'function') {
            highlightStep('preprocessing');
        }

        const result = await apiRequest('/api/recognize', {
            method: 'POST',
            body: formData,
        });

        const elapsed = performance.now() - startTime;

        if (result) {
            // Highlight result steps
            if (typeof highlightStep === 'function') {
                highlightStep('results');
            }

            // Render results
            if (typeof renderResults === 'function') {
                renderResults(result);
            }

            // Show processing time
            if (typeof showProcessingTime === 'function') {
                showProcessingTime(result.processing_time_ms || elapsed);
            }

            setStatus('Recognition complete', 'success');
        }
        // If result is null, apiRequest already handled the error display
    } finally {
        showLoading(false);
        unlock();
    }
}
