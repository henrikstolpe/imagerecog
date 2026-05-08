/**
 * IR Signature Recognition System - Results Module
 *
 * Renders classification and similarity results into the right panel.
 * Handles threshold alerts, color mapping, and result ordering.
 *
 * Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5
 */

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

function getClassificationBars() {
    return document.querySelector('#classification-results .classification-bars');
}

function getSimilarityCards() {
    return document.querySelector('#similarity-results .similarity-cards');
}

function getClassificationAlert() {
    return document.getElementById('classification-alert');
}

function getSimilarityAlert() {
    return document.getElementById('similarity-alert');
}

// ---------------------------------------------------------------------------
// Color Mapping
// ---------------------------------------------------------------------------

/**
 * Map a score to a CSS color class.
 * Green (≥0.7), Yellow ([0.3, 0.7)), Red (<0.3)
 * @param {number} score - Confidence or similarity score
 * @returns {string} CSS class name
 */
function getScoreClass(score) {
    if (score >= 0.7) return 'score-high';
    if (score >= 0.3) return 'score-medium';
    return 'score-low';
}

// ---------------------------------------------------------------------------
// Classification Rendering
// ---------------------------------------------------------------------------

/**
 * Render top-3 classification bars into the classification panel.
 * @param {Array<{vehicle_class: string, confidence: number}>} classifications
 */
function renderClassifications(classifications) {
    const container = getClassificationBars();
    if (!container) return;

    container.innerHTML = '';

    // Take top-3 (should already be sorted by confidence descending from API)
    const top3 = classifications.slice(0, 3);

    top3.forEach((item, index) => {
        const bar = document.createElement('div');
        bar.className = 'confidence-bar';

        // Emphasize top-1 result
        if (index === 0) {
            bar.classList.add('top-result');
        }

        const percentage = Math.round(item.confidence * 100);
        const scoreClass = getScoreClass(item.confidence);

        // Bar header with label and value
        const header = document.createElement('div');
        header.className = 'bar-header';

        const label = document.createElement('span');
        label.className = 'bar-label';
        label.textContent = item.vehicle_class;

        const value = document.createElement('span');
        value.className = 'bar-value';
        value.textContent = `${percentage}%`;

        header.appendChild(label);
        header.appendChild(value);

        // Bar track with fill
        const track = document.createElement('div');
        track.className = 'bar-track';

        const fill = document.createElement('div');
        fill.className = `bar-fill ${scoreClass}`;
        fill.style.width = `${item.confidence * 100}%`;

        track.appendChild(fill);

        bar.appendChild(header);
        bar.appendChild(track);
        container.appendChild(bar);
    });
}

/**
 * Show or hide the low-confidence threshold alert.
 * Visible when top-1 confidence < 0.5.
 * @param {boolean} lowConfidence - Whether the low_confidence flag is set
 */
function updateClassificationAlert(lowConfidence) {
    const alert = getClassificationAlert();
    if (!alert) return;

    if (lowConfidence) {
        alert.removeAttribute('hidden');
    } else {
        alert.setAttribute('hidden', '');
    }
}

// ---------------------------------------------------------------------------
// Similarity Rendering
// ---------------------------------------------------------------------------

/**
 * Render top-5 similarity match cards into the similarity panel.
 * @param {Array<{vehicle_class: string, similarity_score: number, image_id: string, thumbnail_url: string}>} matches
 */
function renderSimilarityMatches(matches) {
    const container = getSimilarityCards();
    if (!container) return;

    container.innerHTML = '';

    // Sort by similarity_score descending (ensure correct order)
    const sorted = [...matches].sort((a, b) => b.similarity_score - a.similarity_score);

    // Take top-5
    const top5 = sorted.slice(0, 5);

    top5.forEach((match) => {
        const card = document.createElement('div');
        const scoreClass = getScoreClass(match.similarity_score);
        card.className = `similarity-card ${scoreClass}`;

        // Thumbnail image
        const img = document.createElement('img');
        img.className = 'card-thumbnail';
        img.src = match.thumbnail_url;
        img.alt = `Reference image ${match.image_id}`;

        // Card info container
        const info = document.createElement('div');
        info.className = 'card-info';

        const cardClass = document.createElement('span');
        cardClass.className = 'card-class';
        cardClass.textContent = match.vehicle_class;

        const cardScore = document.createElement('span');
        cardScore.className = `card-score ${scoreClass}`;
        const percentage = Math.round(match.similarity_score * 100);
        cardScore.textContent = `${percentage}%`;

        const cardId = document.createElement('span');
        cardId.className = 'card-id';
        cardId.textContent = match.image_id;

        info.appendChild(cardClass);
        info.appendChild(cardScore);
        info.appendChild(cardId);

        card.appendChild(img);
        card.appendChild(info);
        container.appendChild(card);
    });
}

/**
 * Show or hide the no-match threshold alert.
 * Visible when all similarity scores < 0.3.
 * @param {boolean} noMatch - Whether the no_match flag is set
 */
function updateSimilarityAlert(noMatch) {
    const alert = getSimilarityAlert();
    if (!alert) return;

    if (noMatch) {
        alert.removeAttribute('hidden');
    } else {
        alert.setAttribute('hidden', '');
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Initialize the results module. Called once on page load.
 */
export function initResults() {
    // Ensure alerts are hidden on init
    clearClassificationResults();
    clearSimilarityResults();
}

/**
 * Render full recognition results into the right panel.
 * @param {object} data - RecognitionResponse from the API
 * @param {Array} data.classifications - Top-3 classification items
 * @param {Array} data.similarity_matches - Top-5 similarity match items
 * @param {boolean} data.low_confidence - Whether top-1 confidence < 0.5
 * @param {boolean} data.no_match - Whether all similarity scores < 0.3
 */
export function renderResults(data) {
    if (!data) return;

    // Render classification bars
    if (data.classifications && data.classifications.length > 0) {
        renderClassifications(data.classifications);
    }

    // Update low-confidence alert
    updateClassificationAlert(data.low_confidence === true);

    // Render similarity cards
    if (data.similarity_matches && data.similarity_matches.length > 0) {
        renderSimilarityMatches(data.similarity_matches);
    }

    // Update no-match alert
    updateSimilarityAlert(data.no_match === true);
}

/**
 * Clear classification results and hide the alert.
 */
export function clearClassificationResults() {
    const container = getClassificationBars();
    if (container) {
        container.innerHTML = '';
    }

    const alert = getClassificationAlert();
    if (alert) {
        alert.setAttribute('hidden', '');
    }
}

/**
 * Clear similarity results and hide the alert.
 */
export function clearSimilarityResults() {
    const container = getSimilarityCards();
    if (container) {
        container.innerHTML = '';
    }

    const alert = getSimilarityAlert();
    if (alert) {
        alert.setAttribute('hidden', '');
    }
}
