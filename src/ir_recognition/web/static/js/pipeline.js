/**
 * IR Signature Recognition System - Pipeline Visualization Module
 *
 * Manages the pipeline step highlighting during recognition processing,
 * displays processing time after completion, and provides reset functionality.
 * Model architecture info (PaliGemma 2 with QLoRA, 2048-d embeddings) is
 * rendered as static HTML — no JS needed for that.
 *
 * Requirements: 6.1, 6.2, 6.3, 6.4
 */

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

let pipelineContainer = null;
let timingElement = null;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Initialize the pipeline module.
 * Gets references to the pipeline container and timing element,
 * and ensures all steps start without the active class.
 */
export function initPipeline() {
    pipelineContainer = document.getElementById('pipeline-visualization');
    timingElement = document.getElementById('pipeline-timing');

    // Ensure all steps start in inactive state
    resetPipeline();
}

/**
 * Highlight a specific step in the pipeline diagram.
 * Removes the active class from all steps, then adds it to the step
 * matching the given data-step attribute value.
 *
 * @param {string} stepName - One of: 'input', 'preprocessing', 'classification', 'similarity', 'results'
 */
export function highlightStep(stepName) {
    if (!pipelineContainer) return;

    const steps = pipelineContainer.querySelectorAll('.pipeline-step');

    // Remove active from all steps
    steps.forEach(step => step.classList.remove('active'));

    // Add active to the matching step
    const targetStep = pipelineContainer.querySelector(`.pipeline-step[data-step="${stepName}"]`);
    if (targetStep) {
        targetStep.classList.add('active');
    }
}

/**
 * Reset the pipeline visualization.
 * Removes the active class from all steps and clears the processing time display.
 */
export function resetPipeline() {
    if (pipelineContainer) {
        const steps = pipelineContainer.querySelectorAll('.pipeline-step');
        steps.forEach(step => step.classList.remove('active'));
    }

    if (timingElement) {
        timingElement.textContent = '';
    }
}

/**
 * Display the processing time after recognition completes.
 * Formats as milliseconds for values under 1000ms, or seconds with one
 * decimal place for values >= 1000ms.
 *
 * @param {number} ms - Processing time in milliseconds
 */
export function showProcessingTime(ms) {
    if (!timingElement) return;

    if (ms >= 1000) {
        timingElement.textContent = `Processing time: ${(ms / 1000).toFixed(1)}s`;
    } else {
        timingElement.textContent = `Processing time: ${Math.round(ms)}ms`;
    }
}
