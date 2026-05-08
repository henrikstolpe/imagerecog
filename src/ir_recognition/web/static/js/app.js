/**
 * Vehicle Recognition System - Main Application Module
 *
 * Entry point for the web UI. Provides shared utilities (status, error display,
 * loading, request locking) and the API request helper with timeout and retry.
 *
 * Requirements: 7.1, 7.2, 7.3, 7.4
 */

import { initUpload } from './upload.js';
import { initResults, clearClassificationResults, clearSimilarityResults } from './results.js';
import { initPipeline } from './pipeline.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _locked = false;
let _lastFailedRequest = null; // { url, options } for retry

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

function getStatusBar() {
    return document.getElementById('status-bar');
}

function getLoadingIndicator() {
    return document.getElementById('loading-indicator');
}

// ---------------------------------------------------------------------------
// Shared Utilities
// ---------------------------------------------------------------------------

/**
 * Update the status bar text and visual type.
 * @param {string} message - Status message to display
 * @param {'ready'|'error'|'success'} type - Visual style class
 */
function setStatus(message, type = 'ready') {
    const bar = getStatusBar();
    if (!bar) return;

    const textEl = bar.querySelector('.status-text');
    if (textEl) {
        textEl.textContent = message;
    }

    // Remove previous type classes and apply new one
    bar.classList.remove('status-ready', 'status-error', 'status-success');
    bar.classList.add(`status-${type}`);

    // Remove any existing retry button when status changes to non-error
    if (type !== 'error') {
        const retryBtn = bar.querySelector('.retry-btn');
        if (retryBtn) retryBtn.remove();
    }
}

/**
 * Display an error message in the status bar and clear stale results.
 * @param {string} message - Error message to display
 */
function showError(message) {
    clearResults();
    setStatus(message, 'error');
}

/**
 * Clear classification and similarity result areas, hiding alerts.
 */
function clearResults() {
    // Clear classification bars
    const classificationBars = document.querySelector('#classification-results .classification-bars');
    if (classificationBars) {
        classificationBars.innerHTML = '';
    }

    // Clear similarity cards
    const similarityCards = document.querySelector('#similarity-results .similarity-cards');
    if (similarityCards) {
        similarityCards.innerHTML = '';
    }

    // Hide alerts
    const classificationAlert = document.getElementById('classification-alert');
    if (classificationAlert) {
        classificationAlert.hidden = true;
    }

    const similarityAlert = document.getElementById('similarity-alert');
    if (similarityAlert) {
        similarityAlert.hidden = true;
    }
}

/**
 * Toggle loading indicator visibility.
 * @param {boolean} show - Whether to show or hide the loading indicator
 */
function showLoading(show) {
    const indicator = getLoadingIndicator();
    if (!indicator) return;
    indicator.hidden = !show;
}

/**
 * Check if a request is currently in progress (locked).
 * @returns {boolean}
 */
function isLocked() {
    return _locked;
}

/**
 * Lock the UI to prevent duplicate submissions.
 */
function lock() {
    _locked = true;
    // Disable interactive submission elements
    const generateBtn = document.getElementById('generate-btn');
    if (generateBtn) generateBtn.disabled = true;

    const dropzone = document.getElementById('upload-dropzone');
    if (dropzone) dropzone.setAttribute('aria-disabled', 'true');
}

/**
 * Unlock the UI after a request completes.
 */
function unlock() {
    _locked = false;
    // Re-enable interactive submission elements
    const generateBtn = document.getElementById('generate-btn');
    if (generateBtn) generateBtn.disabled = false;

    const dropzone = document.getElementById('upload-dropzone');
    if (dropzone) dropzone.removeAttribute('aria-disabled');
}

// ---------------------------------------------------------------------------
// Retry Mechanism
// ---------------------------------------------------------------------------

/**
 * Show a retry button in the status bar for the last failed request.
 */
function showRetryButton() {
    const bar = getStatusBar();
    if (!bar) return;

    // Don't add duplicate retry buttons
    if (bar.querySelector('.retry-btn')) return;

    const retryBtn = document.createElement('button');
    retryBtn.className = 'retry-btn';
    retryBtn.textContent = 'Retry';
    retryBtn.setAttribute('aria-label', 'Retry the last failed request');
    retryBtn.addEventListener('click', handleRetry);
    bar.appendChild(retryBtn);
}

/**
 * Handle retry button click - re-execute the last failed request.
 */
async function handleRetry() {
    if (!_lastFailedRequest) return;

    const { url, options } = _lastFailedRequest;
    _lastFailedRequest = null;

    // Remove retry button
    const bar = getStatusBar();
    const retryBtn = bar ? bar.querySelector('.retry-btn') : null;
    if (retryBtn) retryBtn.remove();

    setStatus('Retrying...', 'ready');
    // Re-issue the request
    await apiRequest(url, options);
}

// ---------------------------------------------------------------------------
// API Request Helper
// ---------------------------------------------------------------------------

const REQUEST_TIMEOUT_MS = 30000;

/**
 * Make an API request with timeout, error handling, and retry support.
 * @param {string} url - The API endpoint URL
 * @param {RequestInit} [options={}] - Fetch options (method, body, headers, etc.)
 * @returns {Promise<any>} Parsed JSON response on success, or null on error
 */
async function apiRequest(url, options = {}) {
    // Abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    // Store for potential retry
    _lastFailedRequest = { url, options };

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            // Parse error detail from response body
            let errorMessage = `Server error (${response.status})`;
            try {
                const errorData = await response.json();
                if (errorData.detail) {
                    errorMessage = errorData.detail;
                }
            } catch {
                // If JSON parsing fails, use status text
                errorMessage = `Server error: ${response.statusText || response.status}`;
            }

            showError(errorMessage);
            showRetryButton();
            return null;
        }

        // Success - clear the stored failed request
        _lastFailedRequest = null;
        return await response.json();
    } catch (error) {
        clearTimeout(timeoutId);

        if (error.name === 'AbortError') {
            // Timeout
            showError('Request timed out. The server took too long to respond.');
            showRetryButton();
            return null;
        }

        // Network / connection error
        showError('Cannot connect to server. Please check your connection.');
        showRetryButton();
        return null;
    }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    // Set initial status
    setStatus('Ready', 'ready');

    // Initialize sub-modules
    initUpload();
    initResults();
    initPipeline();
});

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

export { setStatus, showError, clearResults, showLoading, isLocked, lock, unlock, apiRequest };
