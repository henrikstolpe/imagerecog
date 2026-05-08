/**
 * IR Signature Recognition System - Generator Module
 *
 * Handles synthetic IR image generation controls: vehicle type selection,
 * azimuth/elevation sliders with real-time labels, ambient temperature input,
 * generate button that POSTs to /api/generate, and "Recognize" action on
 * generated images.
 *
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.7
 */

import { apiRequest, showLoading, showError, isLocked, lock, unlock, setStatus } from './app.js';
import { submitForRecognition } from './upload.js';

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

let vehicleTypeSelect = null;
let azimuthSlider = null;
let azimuthValue = null;
let elevationSlider = null;
let elevationValue = null;
let ambientTempInput = null;
let generateBtn = null;
let imagePreview = null;

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/**
 * Initialize the generator module: get DOM references and wire event listeners.
 */
export function initGenerator() {
    vehicleTypeSelect = document.getElementById('vehicle-type-select');
    azimuthSlider = document.getElementById('azimuth-slider');
    azimuthValue = document.getElementById('azimuth-value');
    elevationSlider = document.getElementById('elevation-slider');
    elevationValue = document.getElementById('elevation-value');
    ambientTempInput = document.getElementById('ambient-temp-input');
    generateBtn = document.getElementById('generate-btn');
    imagePreview = document.getElementById('image-preview');

    if (!generateBtn) return;

    // Wire slider real-time label updates
    if (azimuthSlider && azimuthValue) {
        azimuthSlider.addEventListener('input', handleAzimuthChange);
    }

    if (elevationSlider && elevationValue) {
        elevationSlider.addEventListener('input', handleElevationChange);
    }

    // Wire generate button
    generateBtn.addEventListener('click', handleGenerate);
}

// ---------------------------------------------------------------------------
// Slider Handlers
// ---------------------------------------------------------------------------

/**
 * Update azimuth label when slider value changes.
 * @param {Event} event
 */
function handleAzimuthChange(event) {
    const value = event.target.value;
    azimuthValue.textContent = `${value}°`;
    azimuthSlider.setAttribute('aria-valuenow', value);
}

/**
 * Update elevation label when slider value changes.
 * @param {Event} event
 */
function handleElevationChange(event) {
    const value = event.target.value;
    elevationValue.textContent = `${value}°`;
    elevationSlider.setAttribute('aria-valuenow', value);
}

// ---------------------------------------------------------------------------
// Generate Handler
// ---------------------------------------------------------------------------

/**
 * Handle generate button click: read parameters, POST to /api/generate,
 * display the result image with parameters and a recognize button.
 */
async function handleGenerate() {
    if (isLocked()) return;

    lock();
    showLoading(true);
    setStatus('Generating synthetic IR image...', 'ready');

    // Read parameter values from controls
    const vehicleType = vehicleTypeSelect ? vehicleTypeSelect.value : 'T-72';
    const azimuth = azimuthSlider ? parseFloat(azimuthSlider.value) : 0;
    const elevation = elevationSlider ? parseFloat(elevationSlider.value) : 15;
    const ambientTemp = ambientTempInput ? parseFloat(ambientTempInput.value) : 293;

    const requestBody = {
        vehicle_type: vehicleType,
        azimuth: azimuth,
        elevation: elevation,
        ambient_temp: ambientTemp,
    };

    try {
        const result = await apiRequest('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
        });

        if (result) {
            displayGeneratedImage(result);
            setStatus('Image generated successfully', 'success');
        }
        // If result is null, apiRequest already handled the error display
    } finally {
        showLoading(false);
        unlock();
    }
}

// ---------------------------------------------------------------------------
// Display Generated Image
// ---------------------------------------------------------------------------

/**
 * Display the generated image in the preview area with parameters and
 * a "Recognize This Image" button.
 *
 * @param {object} result - The GenerateResponse from the API
 * @param {string} result.image_base64 - Base64-encoded PNG image
 * @param {string} result.vehicle_type - Vehicle type used for generation
 * @param {number} result.azimuth - Azimuth angle used
 * @param {number} result.elevation - Elevation angle used
 * @param {number} result.ambient_temp - Ambient temperature used
 */
function displayGeneratedImage(result) {
    if (!imagePreview) return;

    imagePreview.innerHTML = '';

    // Create image element
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${result.image_base64}`;
    img.alt = `Generated IR image of ${result.vehicle_type} at azimuth ${result.azimuth}°, elevation ${result.elevation}°`;
    img.className = 'preview-image';
    imagePreview.appendChild(img);

    // Create generation parameters display
    const paramsDiv = document.createElement('div');
    paramsDiv.className = 'generation-params';
    paramsDiv.innerHTML = `
        <h4>Generation Parameters</h4>
        <ul>
            <li><strong>Vehicle:</strong> ${escapeHtml(result.vehicle_type)}</li>
            <li><strong>Azimuth:</strong> ${result.azimuth}°</li>
            <li><strong>Elevation:</strong> ${result.elevation}°</li>
            <li><strong>Temperature:</strong> ${result.ambient_temp} K</li>
        </ul>
    `;
    imagePreview.appendChild(paramsDiv);

    // Create "Recognize This Image" button
    const recognizeBtn = document.createElement('button');
    recognizeBtn.type = 'button';
    recognizeBtn.className = 'recognize-generated-btn';
    recognizeBtn.textContent = 'Recognize This Image';
    recognizeBtn.setAttribute('aria-label', 'Submit the generated image for recognition');
    recognizeBtn.addEventListener('click', () => {
        recognizeGeneratedImage(result.image_base64);
    });
    imagePreview.appendChild(recognizeBtn);
}

// ---------------------------------------------------------------------------
// Recognize Generated Image
// ---------------------------------------------------------------------------

/**
 * Convert a base64 PNG image to a File object and submit it for recognition.
 * @param {string} base64Data - Base64-encoded PNG image data
 */
function recognizeGeneratedImage(base64Data) {
    // Decode base64 to binary
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    // Create a File object from the binary data
    const blob = new Blob([bytes], { type: 'image/png' });
    const file = new File([blob], 'generated_ir_image.png', { type: 'image/png' });

    // Submit for recognition using the upload module's function
    submitForRecognition(file);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/**
 * Escape HTML special characters to prevent XSS.
 * @param {string} str - String to escape
 * @returns {string} Escaped string
 */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
