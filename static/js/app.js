/**
 * Plant Disease Classifier - Frontend Application
 *
 * This module handles:
 * - Image upload and preview
 * - API communication with error handling
 * - Results display and formatting
 * - Toast notifications
 */

// ============================================
// Configuration
// ============================================
const API_BASE = '/api/v1';
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];

// ============================================
// State Management
// ============================================
const state = {
    mode: 'full',
    selectedFiles: [],
    isLoading: false,
    currentTreatmentTab: 'organic',
    treatmentData: null,
    selectedModels: ['internal', 'mobilenet_v2'],  // Default selected models for comparison
    selectedAnalysisModel: 'internal',  // Selected model for full analysis
    apiKeys: {
        plantnet: null,
        kindwise: null
    }
};

// ============================================
// DOM Elements
// ============================================
const elements = {
    // Tabs
    tabs: document.querySelectorAll('.tab'),

    // Upload
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    batchFileInput: document.getElementById('batchFileInput'),
    previewArea: document.getElementById('previewArea'),
    previewImages: document.getElementById('previewImages'),
    imageCount: document.getElementById('imageCount'),
    clearBtn: document.getElementById('clearBtn'),

    // Options
    optionsPanel: document.getElementById('optionsPanel'),
    includeTreatment: document.getElementById('includeTreatment'),
    includeExplainability: document.getElementById('includeExplainability'),
    regionSelect: document.getElementById('regionSelect'),
    cropOption: document.getElementById('cropOption'),
    cropSelect: document.getElementById('cropSelect'),

    // Analyze
    analyzeBtn: document.getElementById('analyzeBtn'),

    // Results
    resultsSection: document.getElementById('resultsSection'),
    resultsMeta: document.getElementById('resultsMeta'),
    singleResult: document.getElementById('singleResult'),
    batchResults: document.getElementById('batchResults'),

    // Error
    errorContainer: document.getElementById('errorContainer'),
    errorMessage: document.getElementById('errorMessage'),
    retryBtn: document.getElementById('retryBtn'),

    // API Status
    apiStatus: document.getElementById('apiStatus'),
    apiModels: document.getElementById('apiModels'),

    // Model Selection (Compare Mode)
    modelSelectionPanel: document.getElementById('modelSelectionPanel'),
    modelInternal: document.getElementById('modelInternal'),
    modelMobileNet: document.getElementById('modelMobileNet'),
    modelViT: document.getElementById('modelViT'),
    modelPlantNet: document.getElementById('modelPlantNet'),
    modelResNet50: document.getElementById('modelResNet50'),
    modelEfficientNet: document.getElementById('modelEfficientNet'),
    modelKindwise: document.getElementById('modelKindwise'),

    // Analysis Model (Full Analysis Mode)
    fullAnalysisModelPanel: document.getElementById('fullAnalysisModelPanel'),
    analysisModelSelect: document.getElementById('analysisModelSelect'),

    // API Key inputs
    plantnetApiKey: document.getElementById('plantnetApiKey'),
    kindwiseApiKey: document.getElementById('kindwiseApiKey'),

    // Early Warning Mode
    earlyWarningPanel: document.getElementById('earlyWarningPanel'),
    ewPlantnetApiKey: document.getElementById('ewPlantnetApiKey'),
    ewKindwiseApiKey: document.getElementById('ewKindwiseApiKey'),

    // Early Warning Results
    earlyWarningResults: document.getElementById('earlyWarningResults'),
    severityBanner: document.getElementById('severityBanner'),
    severityIndicator: document.getElementById('severityIndicator'),
    severityScore: document.getElementById('severityScore'),
    severityLabel: document.getElementById('severityLabel'),
    severityUrgency: document.getElementById('severityUrgency'),
    severityTimeline: document.getElementById('severityTimeline'),
    consensusContent: document.getElementById('consensusContent'),
    modelPredictionsGrid: document.getElementById('modelPredictionsGrid'),
    treatmentContentEW: document.getElementById('treatmentContentEW'),
    treatmentMetaEW: document.getElementById('treatmentMetaEW'),
    factorsList: document.getElementById('factorsList'),
    ewMeta: document.getElementById('ewMeta'),

    // Comparison Results
    comparisonResults: document.getElementById('comparisonResults'),
    agreementScore: document.getElementById('agreementScore'),
    comparisonRecommendation: document.getElementById('comparisonRecommendation'),
    comparisonGrid: document.getElementById('comparisonGrid'),
    comparisonMeta: document.getElementById('comparisonMeta'),

    // Toast
    toastContainer: document.getElementById('toastContainer'),
};

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initUpload();
    initOptions();
    initTreatmentTabs();
    initApiKeyInputs();
    checkAPIHealth();

    // Retry button
    elements.retryBtn?.addEventListener('click', handleAnalyze);
});

// ============================================
// API Key Management
// ============================================
function initApiKeyInputs() {
    // Load saved API keys from localStorage
    const savedPlantnetKey = localStorage.getItem('plantnet_api_key');
    const savedKindwiseKey = localStorage.getItem('kindwise_api_key');

    if (savedPlantnetKey && elements.plantnetApiKey) {
        elements.plantnetApiKey.value = savedPlantnetKey;
        state.apiKeys.plantnet = savedPlantnetKey;
    }
    if (savedKindwiseKey && elements.kindwiseApiKey) {
        elements.kindwiseApiKey.value = savedKindwiseKey;
        state.apiKeys.kindwise = savedKindwiseKey;
    }

    // Save API keys on change
    elements.plantnetApiKey?.addEventListener('change', (e) => {
        state.apiKeys.plantnet = e.target.value;
        if (e.target.value) {
            localStorage.setItem('plantnet_api_key', e.target.value);
        } else {
            localStorage.removeItem('plantnet_api_key');
        }
    });

    elements.kindwiseApiKey?.addEventListener('change', (e) => {
        state.apiKeys.kindwise = e.target.value;
        if (e.target.value) {
            localStorage.setItem('kindwise_api_key', e.target.value);
        } else {
            localStorage.removeItem('kindwise_api_key');
        }
    });

    // Analysis model selection
    elements.analysisModelSelect?.addEventListener('change', (e) => {
        state.selectedAnalysisModel = e.target.value;
    });
}

// ============================================
// Tab Management
// ============================================
function initTabs() {
    elements.tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const mode = tab.dataset.mode;
            setMode(mode);
        });
    });
}

function setMode(mode) {
    console.log('setMode called with:', mode);
    state.mode = mode;

    // Update tab styles
    elements.tabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.mode === mode);
    });

    // Show/hide crop option for disease mode
    if (elements.cropOption) {
        elements.cropOption.style.display = mode === 'disease' ? 'flex' : 'none';
    }

    // Show/hide model selection for compare mode
    if (elements.modelSelectionPanel) {
        elements.modelSelectionPanel.style.display = mode === 'compare' ? 'block' : 'none';
        console.log('modelSelectionPanel display:', elements.modelSelectionPanel.style.display);
    } else {
        console.warn('modelSelectionPanel element not found!');
    }

    // Show/hide full analysis model panel
    if (elements.fullAnalysisModelPanel) {
        elements.fullAnalysisModelPanel.style.display = (mode === 'full' || mode === 'species' || mode === 'disease') ? 'flex' : 'none';
    }

    // Show/hide early warning panel
    if (elements.earlyWarningPanel) {
        elements.earlyWarningPanel.style.display = mode === 'early-warning' ? 'block' : 'none';
    }

    // Update file input for batch mode
    if (mode === 'batch') {
        elements.fileInput.removeAttribute('hidden');
        elements.batchFileInput.setAttribute('hidden', '');
    }

    // Clear previous selection when switching modes
    clearSelection();

    // Hide results
    if (elements.resultsSection) {
        elements.resultsSection.style.display = 'none';
    }
}

// ============================================
// File Upload
// ============================================
function initUpload() {
    // Click to upload
    elements.uploadArea.addEventListener('click', () => {
        if (state.mode === 'batch') {
            elements.batchFileInput.click();
        } else {
            elements.fileInput.click();
        }
    });

    // File input change
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.batchFileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);

    // Clear button
    elements.clearBtn.addEventListener('click', clearSelection);

    // Analyze button
    elements.analyzeBtn.addEventListener('click', handleAnalyze);
}

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');

    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    processFiles(files);
    e.target.value = ''; // Reset input
}

function processFiles(files) {
    // Filter valid files
    const validFiles = files.filter(file => {
        if (!ALLOWED_TYPES.includes(file.type)) {
            showToast('error', 'Invalid File Type', `${file.name} is not a valid image file`);
            return false;
        }
        if (file.size > MAX_FILE_SIZE) {
            showToast('error', 'File Too Large', `${file.name} exceeds 10MB limit`);
            return false;
        }
        return true;
    });

    if (validFiles.length === 0) return;

    // Check batch limits
    if (state.mode === 'batch') {
        if (state.selectedFiles.length + validFiles.length > 10) {
            showToast('warning', 'Batch Limit', 'Maximum 10 images allowed');
            validFiles.splice(10 - state.selectedFiles.length);
        }
        state.selectedFiles.push(...validFiles);
    } else {
        state.selectedFiles = [validFiles[0]];
    }

    updatePreview();
}

function updatePreview() {
    if (state.selectedFiles.length === 0) {
        elements.previewArea.style.display = 'none';
        elements.analyzeBtn.disabled = true;
        return;
    }

    elements.previewArea.style.display = 'block';
    elements.analyzeBtn.disabled = false;

    // Update count
    const count = state.selectedFiles.length;
    elements.imageCount.textContent = count > 1 ? `s (${count})` : '';

    // Render previews
    elements.previewImages.innerHTML = '';
    state.selectedFiles.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const div = document.createElement('div');
            div.className = 'preview-image';
            div.innerHTML = `
                <img src="${e.target.result}" alt="Preview ${index + 1}">
                <button class="remove-btn" data-index="${index}">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            `;
            elements.previewImages.appendChild(div);

            // Remove button handler
            div.querySelector('.remove-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                removeFile(index);
            });
        };
        reader.readAsDataURL(file);
    });
}

function removeFile(index) {
    state.selectedFiles.splice(index, 1);
    updatePreview();
}

function clearSelection() {
    state.selectedFiles = [];
    updatePreview();
    elements.resultsSection.style.display = 'none';
}

// ============================================
// Options
// ============================================
function initOptions() {
    // Options are handled via the checkboxes and selects
}

function getOptions() {
    return {
        include_treatment: elements.includeTreatment.checked,
        include_explainability: elements.includeExplainability.checked,
        region: elements.regionSelect.value || null,
        crop: elements.cropSelect.value || null,
    };
}

// ============================================
// API Communication
// ============================================
async function handleAnalyze() {
    if (state.selectedFiles.length === 0 || state.isLoading) return;

    setLoading(true);
    hideError();

    try {
        const options = getOptions();
        let result;
        const selectedModel = state.selectedAnalysisModel || 'internal';

        if (state.mode === 'batch') {
            result = await classifyBatch(state.selectedFiles, options);
            displayBatchResults(result);
        } else if (state.mode === 'early-warning') {
            console.log('Early Warning mode - starting comprehensive analysis');
            const file = state.selectedFiles[0];
            result = await runEarlyWarningAnalysis(file, options);
            displayEarlyWarningResults(result);
        } else if (state.mode === 'compare') {
            console.log('Compare mode - starting comparison');
            const file = state.selectedFiles[0];
            const selectedModels = getSelectedModels();

            if (selectedModels.length === 0) {
                throw new Error('Please select at least one model to compare');
            }

            console.log('Calling classifyCompare with models:', selectedModels);
            result = await classifyCompare(file, selectedModels);
            console.log('Compare API result:', result);
            displayComparisonResults(result);
        } else if (selectedModel !== 'internal' && (state.mode === 'full' || state.mode === 'species' || state.mode === 'disease')) {
            // Use comparison endpoint with single model when non-internal model selected
            const file = state.selectedFiles[0];
            console.log('Using selected model:', selectedModel);
            result = await classifyCompare(file, [selectedModel]);
            displaySingleModelResult(result, selectedModel);
        } else {
            const file = state.selectedFiles[0];

            switch (state.mode) {
                case 'full':
                    result = await classifyFull(file, options);
                    displayFullResults(result);
                    break;
                case 'species':
                    result = await classifySpecies(file, options);
                    displaySpeciesResults(result);
                    break;
                case 'disease':
                    result = await classifyDisease(file, options);
                    displayDiseaseResults(result);
                    break;
            }
        }

        elements.resultsSection.style.display = 'block';
        showToast('success', 'Analysis Complete', 'Your plant image has been analyzed');

    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message || 'An unexpected error occurred');
        showToast('error', 'Analysis Failed', error.message || 'Please try again');
    } finally {
        setLoading(false);
    }
}

async function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            // Remove data URL prefix
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function classifyFull(file, options) {
    const base64 = await fileToBase64(file);

    const response = await fetch(`${API_BASE}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: base64,
            region: options.region,
            include_treatment: options.include_treatment,
            include_explainability: options.include_explainability,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || error.message || `Server error: ${response.status}`);
    }

    return response.json();
}

async function classifySpecies(file, options) {
    const base64 = await fileToBase64(file);

    const response = await fetch(`${API_BASE}/classify/species`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: base64,
            region: options.region,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || error.message || `Server error: ${response.status}`);
    }

    return response.json();
}

async function classifyDisease(file, options) {
    const base64 = await fileToBase64(file);

    const response = await fetch(`${API_BASE}/classify/disease`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: base64,
            crop: options.crop,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || error.message || `Server error: ${response.status}`);
    }

    return response.json();
}

async function classifyBatch(files, options) {
    const images = await Promise.all(files.map(fileToBase64));

    const response = await fetch(`${API_BASE}/classify/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            images,
            region: options.region,
            include_treatment: options.include_treatment,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || error.message || `Server error: ${response.status}`);
    }

    return response.json();
}

// ============================================
// Model Comparison Functions
// ============================================
function getSelectedModels() {
    const models = [];
    if (elements.modelInternal?.checked) models.push('internal');
    if (elements.modelMobileNet?.checked) models.push('mobilenet_v2');
    if (elements.modelViT?.checked) models.push('vit_crop');
    if (elements.modelResNet50?.checked) models.push('resnet50_plant');
    if (elements.modelEfficientNet?.checked) models.push('efficientnet_plant');
    if (elements.modelPlantNet?.checked) models.push('plantnet');
    if (elements.modelKindwise?.checked) models.push('kindwise');
    console.log('getSelectedModels:', models);
    return models;
}

async function classifyCompare(file, models) {
    const base64 = await fileToBase64(file);

    // Build request with API keys if provided
    const requestBody = {
        image: base64,
        models: models,
    };

    // Include API keys if provided
    if (state.apiKeys.plantnet) {
        requestBody.plantnet_api_key = state.apiKeys.plantnet;
    }
    if (state.apiKeys.kindwise) {
        requestBody.kindwise_api_key = state.apiKeys.kindwise;
    }

    const response = await fetch(`${API_BASE}/classify/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || error.message || `Server error: ${response.status}`);
    }

    return response.json();
}

// ============================================
// Early Warning System Functions
// ============================================
async function runEarlyWarningAnalysis(file, options) {
    const base64 = await fileToBase64(file);

    // Get API keys from early warning inputs or fallback to compare mode inputs
    const plantnetKey = elements.ewPlantnetApiKey?.value || state.apiKeys.plantnet;
    const kindwiseKey = elements.ewKindwiseApiKey?.value || state.apiKeys.kindwise;

    const requestBody = {
        image: base64,
        region: options.region,
    };

    if (plantnetKey) {
        requestBody.plantnet_api_key = plantnetKey;
    }
    if (kindwiseKey) {
        requestBody.kindwise_api_key = kindwiseKey;
    }

    const response = await fetch(`${API_BASE}/classify/early-warning`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || error.message || `Server error: ${response.status}`);
    }

    return response.json();
}

function displayEarlyWarningResults(data) {
    console.log('displayEarlyWarningResults:', data);

    // Hide other result containers
    if (elements.singleResult) elements.singleResult.style.display = 'none';
    if (elements.batchResults) elements.batchResults.style.display = 'none';
    if (elements.comparisonResults) elements.comparisonResults.style.display = 'none';
    if (elements.errorContainer) elements.errorContainer.style.display = 'none';

    // Show early warning results
    elements.earlyWarningResults.style.display = 'block';

    // Display severity banner
    displaySeverityBanner(data.severity, data.consensus);

    // Display consensus
    displayConsensus(data.consensus);

    // Display model predictions
    displayModelPredictions(data.model_predictions);

    // Display treatment recommendations
    displayEWTreatment(data.treatment);
    initEWTreatmentTabs(data.treatment);

    // Display severity factors
    displaySeverityFactors(data.severity);

    // Display meta with fallback indicators
    if (elements.ewMeta) {
        const severityFallback = data.metadata?.severity_is_fallback;
        const treatmentFallback = data.metadata?.treatment_is_fallback;
        const treatmentData = data.metadata?.treatment_data || {};

        elements.ewMeta.innerHTML = `
            <div class="ew-meta-info">
                <span>Analysis completed in ${data.metadata?.total_processing_time_ms?.toFixed(0) || '?'}ms</span>
                <span>Models consulted: ${data.metadata?.models_consulted || '?'}</span>
                ${data.metadata?.region ? `<span>Region: ${data.metadata.region}</span>` : ''}
                ${treatmentData.loaded ? `<span>Treatment database: ${treatmentData.disease_count} diseases</span>` : ''}
            </div>
            ${(severityFallback || treatmentFallback) ? `
                <div class="ew-meta-fallback">
                    ${severityFallback ? `
                        <span class="fallback-badge">
                            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                                <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                            </svg>
                            Severity: Fallback
                        </span>
                    ` : ''}
                    ${treatmentFallback ? `
                        <span class="fallback-badge">
                            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                                <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                            </svg>
                            Treatment: Fallback
                        </span>
                    ` : ''}
                </div>
            ` : ''}
        `;
    }
}

function displaySeverityBanner(severity, consensus) {
    const banner = elements.severityBanner;
    if (!banner) return;

    // Set severity class
    banner.className = `severity-banner severity-${severity.level}`;

    // Update score
    elements.severityScore.textContent = Math.round(severity.score);
    elements.severityLabel.textContent = severity.level.toUpperCase();

    // Update urgency info
    elements.severityUrgency.textContent = consensus.is_healthy
        ? 'Plant Appears Healthy'
        : consensus.disease_name;
    elements.severityTimeline.textContent = severity.urgency;

    // Show fallback indicator in severity section if applicable
    const severitySection = document.getElementById('severitySection');
    if (severitySection) {
        let existingWarning = severitySection.querySelector('.fallback-warning');
        if (existingWarning) existingWarning.remove();

        if (severity.is_fallback) {
            const warningDiv = document.createElement('div');
            warningDiv.className = 'fallback-warning severity-fallback';
            warningDiv.innerHTML = `
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
                <span><strong>FALLBACK:</strong> Severity data for this disease is not in our database. Using default moderate severity (50).</span>
            `;
            banner.insertAdjacentElement('afterend', warningDiv);
        }
    }
}

function displayConsensus(consensus) {
    const container = elements.consensusContent;
    if (!container) return;

    const agreementPercent = (consensus.model_agreement * 100).toFixed(0);
    const confidencePercent = (consensus.confidence * 100).toFixed(0);

    container.innerHTML = `
        <div class="consensus-main">
            <div class="consensus-disease ${consensus.is_healthy ? 'healthy' : 'diseased'}">
                <span class="disease-name">${consensus.disease_name}</span>
                <span class="confidence-badge">${confidencePercent}% confidence</span>
            </div>
            <div class="consensus-agreement">
                <div class="agreement-bar">
                    <div class="agreement-fill" style="width: ${agreementPercent}%"></div>
                </div>
                <span class="agreement-text">${agreementPercent}% model agreement</span>
            </div>
        </div>
        <div class="consensus-reasoning">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="16" x2="12" y2="12"/>
                <line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>
            <p>${consensus.reasoning}</p>
        </div>
        <div class="consensus-models">
            ${consensus.supporting_models.length > 0 ? `
                <div class="models-group supporting">
                    <span class="models-label">Supporting:</span>
                    ${consensus.supporting_models.map(m => `<span class="model-tag support">${m}</span>`).join('')}
                </div>
            ` : ''}
            ${consensus.dissenting_models.length > 0 ? `
                <div class="models-group dissenting">
                    <span class="models-label">Different opinion:</span>
                    ${consensus.dissenting_models.map(m => `<span class="model-tag dissent">${m}</span>`).join('')}
                </div>
            ` : ''}
        </div>
    `;
}

function displayModelPredictions(predictions) {
    const grid = elements.modelPredictionsGrid;
    if (!grid) return;

    grid.innerHTML = predictions.map(pred => {
        const hasError = pred.error;
        const isHealthy = (pred.prediction || '').toLowerCase().includes('healthy');
        const confidencePercent = ((pred.confidence || 0) * 100).toFixed(1);

        if (hasError) {
            return `
                <div class="model-prediction-card error">
                    <div class="mpc-header">
                        <h4>${pred.model_name}</h4>
                        <span class="mpc-type">${pred.model_type}</span>
                    </div>
                    <div class="mpc-error">
                        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="12" y1="8" x2="12" y2="12"/>
                            <line x1="12" y1="16" x2="12.01" y2="16"/>
                        </svg>
                        <span>${pred.error}</span>
                    </div>
                </div>
            `;
        }

        return `
            <div class="model-prediction-card ${isHealthy ? 'healthy' : 'diseased'}">
                <div class="mpc-header">
                    <h4>${pred.model_name}</h4>
                    <span class="mpc-type">${pred.model_type}</span>
                </div>
                <div class="mpc-prediction">
                    <span class="mpc-disease ${isHealthy ? 'healthy' : 'diseased'}">${pred.prediction || 'Unknown'}</span>
                    <div class="mpc-confidence">
                        <div class="confidence-bar-mini">
                            <div class="confidence-fill-mini" style="width: ${confidencePercent}%"></div>
                        </div>
                        <span>${confidencePercent}%</span>
                    </div>
                </div>
                <div class="mpc-explanation">
                    <p>${pred.explanation}</p>
                </div>
                ${pred.contributing_factors && pred.contributing_factors.length > 0 ? `
                    <div class="mpc-factors">
                        <span class="factors-label">Contributing factors:</span>
                        <ul>
                            ${pred.contributing_factors.map(f => `<li>${f}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                <div class="mpc-time">
                    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <polyline points="12 6 12 12 16 14"/>
                    </svg>
                    ${pred.processing_time_ms?.toFixed(0) || '?'}ms
                </div>
            </div>
        `;
    }).join('');
}

let currentEWTreatmentTab = 'immediate';
let ewTreatmentData = null;

function displayEWTreatment(treatment) {
    ewTreatmentData = treatment;
    displayEWTreatmentTab('immediate');

    // Display fallback warning if applicable
    const treatmentCard = document.getElementById('treatmentCardEW');
    let existingWarning = treatmentCard?.querySelector('.fallback-warning');
    if (existingWarning) existingWarning.remove();

    if (treatment.is_fallback && treatmentCard) {
        const warningDiv = document.createElement('div');
        warningDiv.className = 'fallback-warning treatment-fallback';
        warningDiv.innerHTML = `
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            <span><strong>FALLBACK RESPONSE:</strong> This disease is not in our treatment database. The recommendations shown are general guidelines. Please consult your local agricultural extension service for specific advice.</span>
        `;
        const cardContent = treatmentCard.querySelector('.card-content');
        if (cardContent) {
            cardContent.insertBefore(warningDiv, cardContent.firstChild);
        }
    }

    // Display meta info
    if (elements.treatmentMetaEW) {
        elements.treatmentMetaEW.innerHTML = `
            ${treatment.data_source ? `
                <div class="data-source-info ${treatment.is_fallback ? 'fallback' : ''}">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="16" y1="13" x2="8" y2="13"/>
                        <line x1="16" y1="17" x2="8" y2="17"/>
                        <polyline points="10 9 9 9 8 9"/>
                    </svg>
                    <span><strong>Data Source:</strong> ${treatment.data_source}</span>
                </div>
            ` : ''}
            <div class="treatment-meta-grid">
                <div class="meta-item">
                    <span class="meta-label">Monitoring Schedule</span>
                    <span class="meta-value">${treatment.monitoring_schedule}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Estimated Recovery</span>
                    <span class="meta-value">${treatment.estimated_recovery}</span>
                </div>
                ${treatment.weather_considerations ? `
                    <div class="meta-item full-width">
                        <span class="meta-label">Weather Considerations</span>
                        <span class="meta-value">${treatment.weather_considerations}</span>
                    </div>
                ` : ''}
                ${treatment.regional_notes ? `
                    <div class="meta-item full-width regional">
                        <span class="meta-label">Regional Notes</span>
                        <span class="meta-value">${treatment.regional_notes}</span>
                    </div>
                ` : ''}
            </div>
        `;
    }
}

function initEWTreatmentTabs(treatment) {
    document.querySelectorAll('.treatment-tab-ew').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabType = tab.dataset.treatmentEw;

            // Update active state
            document.querySelectorAll('.treatment-tab-ew').forEach(t =>
                t.classList.toggle('active', t.dataset.treatmentEw === tabType)
            );

            currentEWTreatmentTab = tabType;
            displayEWTreatmentTab(tabType);
        });
    });
}

function displayEWTreatmentTab(tabType) {
    if (!ewTreatmentData || !elements.treatmentContentEW) return;

    const tabMapping = {
        'immediate': 'immediate_actions',
        'organic': 'organic_treatments',
        'chemical': 'chemical_treatments',
        'prevention': 'prevention_measures'
    };

    const items = ewTreatmentData[tabMapping[tabType]] || [];

    if (items.length === 0) {
        elements.treatmentContentEW.innerHTML = `
            <p class="no-treatment">No ${tabType} recommendations available</p>
        `;
        return;
    }

    elements.treatmentContentEW.innerHTML = `
        <div class="treatment-list-ew">
            ${items.map((item, i) => {
                // Check if this is a fallback indicator item
                const isFallbackItem = item.includes('FALLBACK') || item.includes('\u26a0\ufe0f FALLBACK');
                return `
                    <div class="treatment-item-ew ${isFallbackItem ? 'fallback-item' : ''}">
                        <span class="treatment-number">${i + 1}</span>
                        <span class="treatment-text">${item}</span>
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

function displaySeverityFactors(severity) {
    const container = elements.factorsList;
    if (!container) return;

    container.innerHTML = severity.factors.map(factor => `
        <div class="factor-item">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="9 11 12 14 22 4"/>
                <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
            </svg>
            <span>${factor}</span>
        </div>
    `).join('');
}

function displaySingleModelResult(data, modelKey) {
    console.log('displaySingleModelResult called with:', data, modelKey);

    // Get the external model result
    const modelResult = data.external_models?.[modelKey];

    if (!modelResult) {
        showError('No result from selected model');
        return;
    }

    if (modelResult.error) {
        showError(modelResult.error);
        return;
    }

    // Hide other result containers
    if (elements.batchResults) elements.batchResults.style.display = 'none';
    if (elements.comparisonResults) elements.comparisonResults.style.display = 'none';
    if (elements.errorContainer) elements.errorContainer.style.display = 'none';

    // Show single result
    elements.singleResult.style.display = 'grid';

    // Hide treatment and explainability cards (not available for external models)
    document.getElementById('treatmentCard').style.display = 'none';
    document.getElementById('explainCard').style.display = 'none';

    // Meta info
    if (data.metadata) {
        elements.resultsMeta.innerHTML = `
            Model: ${modelResult.model_name} | Processed in ${modelResult.processing_time_ms?.toFixed(0) || '?'}ms
        `;
    }

    // Display plant identification (from additional_info if available)
    const additionalInfo = modelResult.additional_info || {};
    const plantData = {
        family: { name: additionalInfo.family || 'Unknown', confidence: modelResult.confidence || 0 },
        genus: { name: additionalInfo.genus || 'Unknown', confidence: modelResult.confidence || 0 },
        species: { name: modelResult.raw_label || modelResult.prediction || 'Unknown', confidence: modelResult.confidence || 0 },
        common_name: additionalInfo.common_names?.[0] || modelResult.prediction,
        alternative_species: additionalInfo.top_3?.slice(1).map(t => ({
            name: t.disease || t.label || t.species || 'Unknown',
            confidence: t.confidence
        }))
    };
    displayPlantIdentification(plantData);

    // Display health assessment
    const isHealthy = (modelResult.prediction || '').toLowerCase() === 'healthy';
    const healthData = {
        status: isHealthy ? 'Healthy' : 'Diseased',
        disease: isHealthy ? null : modelResult.prediction,
        confidence: modelResult.confidence || 0,
        visual_symptoms: []
    };
    displayHealthAssessment(healthData);
}

function displayComparisonResults(data) {
    console.log('displayComparisonResults called with:', data);

    // Verify elements exist
    if (!elements.comparisonResults) {
        console.error('comparisonResults element not found!');
        return;
    }

    // Hide other result containers
    if (elements.singleResult) elements.singleResult.style.display = 'none';
    if (elements.batchResults) elements.batchResults.style.display = 'none';
    if (elements.errorContainer) elements.errorContainer.style.display = 'none';

    // Show comparison results
    elements.comparisonResults.style.display = 'block';
    console.log('comparisonResults display set to block');

    // Agreement Score
    const scoreContainer = elements.agreementScore;
    if (data.agreement_score !== null && data.agreement_score !== undefined) {
        const scorePercent = (data.agreement_score * 100).toFixed(0);
        const scoreClass = data.agreement_score >= 0.8 ? 'high' :
                          data.agreement_score >= 0.5 ? 'medium' : 'low';
        scoreContainer.innerHTML = `
            <div class="score-circle ${scoreClass}">
                <span class="score-value">${scorePercent}%</span>
                <span class="score-label">Agreement</span>
            </div>
        `;
    } else {
        scoreContainer.innerHTML = `
            <div class="score-circle unknown">
                <span class="score-value">N/A</span>
                <span class="score-label">Agreement</span>
            </div>
        `;
    }

    // Recommendation
    elements.comparisonRecommendation.innerHTML = `
        <div class="recommendation-text">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="16" x2="12" y2="12"/>
                <line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>
            ${data.recommendation || 'No recommendation available'}
        </div>
    `;

    // Build comparison grid
    const grid = elements.comparisonGrid;
    grid.innerHTML = '';

    // Internal model card
    if (data.internal) {
        grid.innerHTML += createModelCard('Internal Model', data.internal, true);
    }

    // External model cards
    if (data.external_models) {
        for (const [modelKey, result] of Object.entries(data.external_models)) {
            grid.innerHTML += createModelCard(result.model_name || modelKey, result, false);
        }
    }

    // Meta info
    if (data.metadata) {
        elements.comparisonMeta.innerHTML = `
            <div class="comparison-meta-info">
                <span>Total processing time: ${data.metadata.total_processing_time_ms?.toFixed(0) || '?'}ms</span>
                <span>Models compared: ${data.metadata.models_compared || '?'}</span>
            </div>
        `;
    }
}

function createModelCard(modelName, result, isInternal) {
    const hasError = result.error;
    const isHealthy = isInternal
        ? result.health_status === 'Healthy'
        : (result.prediction || '').toLowerCase() === 'healthy';

    // Extract prediction info based on model type
    let prediction, confidence, additionalInfo;

    if (isInternal) {
        prediction = result.disease || result.health_status || 'Unknown';
        confidence = result.disease_confidence || result.species_confidence || 0;
        additionalInfo = `
            <div class="model-detail">
                <span class="detail-label">Species:</span>
                <span class="detail-value">${result.species || 'Unknown'}</span>
            </div>
            <div class="model-detail">
                <span class="detail-label">Common Name:</span>
                <span class="detail-value">${result.common_name || 'N/A'}</span>
            </div>
            <div class="model-detail">
                <span class="detail-label">Health Status:</span>
                <span class="detail-value ${isHealthy ? 'healthy' : 'diseased'}">${result.health_status}</span>
            </div>
            ${result.disease ? `
                <div class="model-detail">
                    <span class="detail-label">Disease:</span>
                    <span class="detail-value diseased">${result.disease}</span>
                </div>
            ` : ''}
        `;
    } else {
        prediction = result.prediction || 'Unknown';
        confidence = result.confidence || 0;
        const crop = result.additional_info?.crop || 'N/A';
        const top3 = result.additional_info?.top_3 || [];

        additionalInfo = `
            <div class="model-detail">
                <span class="detail-label">Raw Label:</span>
                <span class="detail-value">${result.raw_label || 'N/A'}</span>
            </div>
            <div class="model-detail">
                <span class="detail-label">Detected Crop:</span>
                <span class="detail-value">${crop}</span>
            </div>
            ${top3.length > 0 ? `
                <div class="model-alternatives">
                    <span class="detail-label">Top Predictions:</span>
                    <ol class="alt-list">
                        ${top3.map(t => `
                            <li>
                                <span class="alt-name">${t.disease || t.label || t.species || 'Unknown'}</span>
                                <span class="alt-conf">${(t.confidence * 100).toFixed(1)}%</span>
                            </li>
                        `).join('')}
                    </ol>
                </div>
            ` : ''}
        `;
    }

    if (hasError) {
        return `
            <div class="model-card error">
                <div class="model-card-header">
                    <h4>${modelName}</h4>
                    <span class="model-badge ${isInternal ? 'internal' : 'external'}">
                        ${isInternal ? 'Internal' : 'External'}
                    </span>
                </div>
                <div class="model-card-body">
                    <div class="model-error">
                        <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="12" y1="8" x2="12" y2="12"/>
                            <line x1="12" y1="16" x2="12.01" y2="16"/>
                        </svg>
                        <span>${result.error}</span>
                    </div>
                </div>
            </div>
        `;
    }

    return `
        <div class="model-card ${isHealthy ? 'healthy' : 'diseased'}">
            <div class="model-card-header">
                <h4>${modelName}</h4>
                <span class="model-badge ${isInternal ? 'internal' : 'external'}">
                    ${isInternal ? 'Internal' : 'External'}
                </span>
            </div>
            <div class="model-card-body">
                <div class="model-prediction">
                    <span class="prediction-label">Prediction:</span>
                    <span class="prediction-value ${isHealthy ? 'healthy' : 'diseased'}">${prediction}</span>
                </div>
                <div class="model-confidence">
                    <div class="confidence-bar-container">
                        <div class="confidence-bar-fill" style="width: ${confidence * 100}%"></div>
                    </div>
                    <span class="confidence-text">${(confidence * 100).toFixed(1)}% confidence</span>
                </div>
                <div class="model-details">
                    ${additionalInfo}
                </div>
                <div class="model-time">
                    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <polyline points="12 6 12 12 16 14"/>
                    </svg>
                    ${result.processing_time_ms?.toFixed(0) || '?'}ms
                </div>
            </div>
        </div>
    `;
}

// ============================================
// Results Display
// ============================================
function displayFullResults(data) {
    elements.singleResult.style.display = 'grid';
    elements.batchResults.style.display = 'none';
    elements.errorContainer.style.display = 'none';

    // Meta info
    if (data.metadata) {
        elements.resultsMeta.innerHTML = `
            Processed in ${data.metadata.processing_time_ms?.toFixed(0) || '?'}ms
        `;
    }

    // Plant identification
    displayPlantIdentification(data.plant);

    // Health assessment
    displayHealthAssessment(data.health);

    // Treatment (if available)
    if (data.treatment) {
        displayTreatment(data.treatment);
        document.getElementById('treatmentCard').style.display = 'block';
    } else {
        document.getElementById('treatmentCard').style.display = 'none';
    }

    // Explainability (if available)
    if (data.explainability) {
        displayExplainability(data.explainability);
        document.getElementById('explainCard').style.display = 'block';
    } else {
        document.getElementById('explainCard').style.display = 'none';
    }
}

function displaySpeciesResults(data) {
    elements.singleResult.style.display = 'grid';
    elements.batchResults.style.display = 'none';
    elements.errorContainer.style.display = 'none';

    // Hide cards not relevant for species-only
    document.querySelector('.health-card').style.display = 'none';
    document.getElementById('treatmentCard').style.display = 'none';
    document.getElementById('explainCard').style.display = 'none';
    document.querySelector('.plant-card').style.display = 'block';

    // Meta info
    if (data.metadata) {
        elements.resultsMeta.innerHTML = `
            Processed in ${data.metadata.processing_time_ms?.toFixed(0) || '?'}ms
        `;
    }

    // Convert to expected format
    const plantData = {
        family: data.family,
        genus: data.genus,
        species: data.species,
        common_name: data.common_name,
        alternative_species: data.alternatives?.map(a => ({
            name: a.name,
            confidence: a.confidence
        })),
    };

    displayPlantIdentification(plantData);
}

function displayDiseaseResults(data) {
    elements.singleResult.style.display = 'grid';
    elements.batchResults.style.display = 'none';
    elements.errorContainer.style.display = 'none';

    // Hide cards not relevant for disease-only
    document.querySelector('.plant-card').style.display = 'none';
    document.getElementById('treatmentCard').style.display = 'none';
    document.getElementById('explainCard').style.display = 'none';
    document.querySelector('.health-card').style.display = 'block';

    // Meta info
    if (data.metadata) {
        elements.resultsMeta.innerHTML = `
            Processed in ${data.metadata.processing_time_ms?.toFixed(0) || '?'}ms
        `;
    }

    // Convert to expected format
    const healthData = {
        status: data.status,
        disease: data.disease,
        confidence: data.confidence,
        visual_symptoms: data.visual_symptoms || [],
    };

    displayHealthAssessment(healthData);
}

function displayPlantIdentification(plant) {
    const taxonomyTree = document.getElementById('taxonomyTree');
    const commonName = document.getElementById('commonName');
    const alternatives = document.getElementById('alternatives');

    // Show plant card
    document.querySelector('.plant-card').style.display = 'block';

    // Taxonomy tree
    const levels = [
        { rank: 'Family', data: plant.family },
        { rank: 'Genus', data: plant.genus },
        { rank: 'Species', data: plant.species },
    ];

    taxonomyTree.innerHTML = levels.map(level => `
        <div class="taxonomy-level">
            <span class="taxonomy-rank">${level.rank}</span>
            <span class="taxonomy-name">${level.data?.name || 'Unknown'}</span>
            <div class="taxonomy-confidence">
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${(level.data?.confidence || 0) * 100}%"></div>
                </div>
                <span class="confidence-value">${((level.data?.confidence || 0) * 100).toFixed(0)}%</span>
            </div>
        </div>
    `).join('');

    // Common name
    if (plant.common_name) {
        commonName.innerHTML = `
            <div class="common-name-label">Common Name</div>
            <div class="common-name-value">${plant.common_name}</div>
        `;
        commonName.style.display = 'block';
    } else {
        commonName.style.display = 'none';
    }

    // Alternatives
    if (plant.alternative_species && plant.alternative_species.length > 0) {
        alternatives.innerHTML = `
            <div class="alternatives-title">Other Possible Matches</div>
            ${plant.alternative_species.map(alt => `
                <div class="alternative-item">
                    <span>${alt.name}</span>
                    <span>${(alt.confidence * 100).toFixed(0)}%</span>
                </div>
            `).join('')}
        `;
        alternatives.style.display = 'block';
    } else {
        alternatives.style.display = 'none';
    }
}

function displayHealthAssessment(health) {
    const healthStatus = document.getElementById('healthStatus');
    const diseaseInfo = document.getElementById('diseaseInfo');
    const symptomsList = document.getElementById('symptomsList');

    // Show health card
    document.querySelector('.health-card').style.display = 'block';

    const isHealthy = health.status === 'Healthy';

    // Health status
    healthStatus.className = `health-status ${isHealthy ? 'healthy' : 'diseased'}`;
    healthStatus.innerHTML = `
        <div class="health-status-icon">
            ${isHealthy ? `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
            ` : `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
            `}
        </div>
        <div class="health-status-text">
            <h4>${health.status}</h4>
            <p>Confidence: ${(health.confidence * 100).toFixed(0)}%</p>
        </div>
    `;

    // Disease info
    if (!isHealthy && health.disease) {
        diseaseInfo.innerHTML = `
            <div class="disease-name">${health.disease}</div>
            ${health.disease_stage ? `
                <div class="disease-stage">
                    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                    </svg>
                    Stage: ${health.disease_stage}
                </div>
            ` : ''}
        `;
        diseaseInfo.style.display = 'block';
    } else {
        diseaseInfo.style.display = 'none';
    }

    // Symptoms
    if (health.visual_symptoms && health.visual_symptoms.length > 0) {
        symptomsList.innerHTML = `
            <h4>Visual Symptoms</h4>
            ${health.visual_symptoms.map(symptom => `
                <div class="symptom-item">${symptom}</div>
            `).join('')}
        `;
        symptomsList.style.display = 'block';
    } else {
        symptomsList.style.display = 'none';
    }
}

function displayTreatment(treatment) {
    state.treatmentData = treatment;

    // Urgency
    const urgency = document.getElementById('treatmentUrgency');
    const urgencyClass = treatment.urgency || 'routine';
    urgency.className = `treatment-urgency ${urgencyClass}`;
    urgency.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <polyline points="12 6 12 12 16 14"/>
        </svg>
        ${urgencyClass.charAt(0).toUpperCase() + urgencyClass.slice(1)} Action Required
    `;

    // Display current tab content
    displayTreatmentTab(state.currentTreatmentTab);

    // Region notes
    const regionNotes = document.getElementById('regionNotes');
    if (treatment.region_specific_notes) {
        regionNotes.innerHTML = `<strong>Note:</strong> ${treatment.region_specific_notes}`;
        regionNotes.style.display = 'block';
    } else {
        regionNotes.style.display = 'none';
    }
}

function initTreatmentTabs() {
    document.querySelectorAll('.treatment-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabType = tab.dataset.treatment;

            // Update active state
            document.querySelectorAll('.treatment-tab').forEach(t =>
                t.classList.toggle('active', t.dataset.treatment === tabType)
            );

            state.currentTreatmentTab = tabType;
            displayTreatmentTab(tabType);
        });
    });
}

function displayTreatmentTab(tabType) {
    if (!state.treatmentData) return;

    const content = document.getElementById('treatmentContent');
    const items = state.treatmentData[tabType] || [];

    if (items.length === 0) {
        content.innerHTML = `<p style="color: var(--color-text-muted); padding: var(--space-md);">No ${tabType} treatments available</p>`;
        return;
    }

    content.innerHTML = `
        <div class="treatment-list">
            ${items.map((item, i) => `
                <div class="treatment-item">
                    <div class="treatment-item-icon">${i + 1}</div>
                    <div>${item}</div>
                </div>
            `).join('')}
        </div>
    `;
}

function displayExplainability(explain) {
    // Reasoning
    const reasoning = document.getElementById('modelReasoning');
    reasoning.textContent = explain.model_reasoning || 'No reasoning available';

    // Confidence notes
    const confidenceNotes = document.getElementById('confidenceNotes');
    confidenceNotes.textContent = explain.confidence_notes || '';

    // Key features
    const keyFeatures = document.getElementById('keyFeatures');
    if (explain.key_features && explain.key_features.length > 0) {
        keyFeatures.innerHTML = `
            <h4>Key Features Detected</h4>
            ${explain.key_features.map(f => `<span class="feature-tag">${f}</span>`).join('')}
        `;
        keyFeatures.style.display = 'block';
    } else {
        keyFeatures.style.display = 'none';
    }

    // Uncertainty factors
    const uncertainty = document.getElementById('uncertaintyFactors');
    if (explain.uncertainty_factors && explain.uncertainty_factors.length > 0) {
        uncertainty.innerHTML = `
            <h4>Uncertainty Factors</h4>
            ${explain.uncertainty_factors.map(f => `
                <div class="uncertainty-item">${f}</div>
            `).join('')}
        `;
        uncertainty.style.display = 'block';
    } else {
        uncertainty.style.display = 'none';
    }

    // Grad-CAM
    const gradCamSection = document.getElementById('gradCamSection');
    if (explain.grad_cam?.heatmap_base64) {
        document.getElementById('gradCamImage').src =
            `data:image/png;base64,${explain.grad_cam.heatmap_base64}`;

        const focusRegions = document.getElementById('focusRegions');
        if (explain.grad_cam.focus_regions?.length > 0) {
            focusRegions.innerHTML = explain.grad_cam.focus_regions.map(r =>
                `<div>${r}</div>`
            ).join('');
        }

        gradCamSection.style.display = 'block';
    } else {
        gradCamSection.style.display = 'none';
    }
}

function displayBatchResults(data) {
    elements.singleResult.style.display = 'none';
    elements.batchResults.style.display = 'block';
    elements.errorContainer.style.display = 'none';

    // Summary
    const summary = document.getElementById('batchSummary');
    const s = data.summary;
    summary.innerHTML = `
        <div class="summary-stat">
            <div class="summary-stat-value">${s.total_images}</div>
            <div class="summary-stat-label">Total Images</div>
        </div>
        <div class="summary-stat">
            <div class="summary-stat-value">${s.successful}</div>
            <div class="summary-stat-label">Processed</div>
        </div>
        <div class="summary-stat healthy">
            <div class="summary-stat-value">${s.healthy_count}</div>
            <div class="summary-stat-label">Healthy</div>
        </div>
        <div class="summary-stat diseased">
            <div class="summary-stat-value">${s.diseased_count}</div>
            <div class="summary-stat-label">Diseased</div>
        </div>
    `;

    // Results grid
    const grid = document.getElementById('batchGrid');
    grid.innerHTML = data.results.map((result, i) => {
        const hasError = result.metadata?.error;
        const species = result.plant?.species?.name || 'Unknown';
        const health = result.health?.status || 'Unknown';
        const isHealthy = health === 'Healthy';

        // Get image from state
        const file = state.selectedFiles[i];
        const imgSrc = file ? URL.createObjectURL(file) : '';

        return `
            <div class="batch-item">
                <img class="batch-item-image" src="${imgSrc}" alt="Image ${i + 1}">
                <div class="batch-item-content">
                    ${hasError ? `
                        <div class="batch-item-error">Error: ${result.metadata.error}</div>
                    ` : `
                        <div class="batch-item-species">${species}</div>
                        <div class="batch-item-health ${isHealthy ? 'healthy' : 'diseased'}">
                            ${health}${result.health?.disease ? `: ${result.health.disease}` : ''}
                        </div>
                    `}
                </div>
            </div>
        `;
    }).join('');
}

// ============================================
// Error Handling
// ============================================
function showError(message) {
    elements.errorContainer.style.display = 'flex';
    elements.singleResult.style.display = 'none';
    elements.batchResults.style.display = 'none';
    elements.errorMessage.textContent = message;
    elements.resultsSection.style.display = 'block';
}

function hideError() {
    elements.errorContainer.style.display = 'none';
}

// ============================================
// Loading State
// ============================================
function setLoading(loading) {
    state.isLoading = loading;
    elements.analyzeBtn.disabled = loading || state.selectedFiles.length === 0;
    elements.analyzeBtn.classList.toggle('loading', loading);
}

// ============================================
// API Health Check
// ============================================
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE}/health/ready`);
        const data = await response.json();

        if (response.ok) {
            elements.apiStatus.classList.add('online');
            elements.apiStatus.classList.remove('offline');
            elements.apiStatus.querySelector('.status-text').textContent = 'API Online';

            // Update models info
            if (data.components) {
                const modelsInfo = elements.apiModels.querySelector('.models-info');
                modelsInfo.innerHTML = `
                    <div><strong>Species Classifier:</strong> ${data.components.species_classifier?.version || 'N/A'}</div>
                    <div><strong>Disease Detector:</strong> ${data.components.disease_detector?.version || 'N/A'}</div>
                    <div><strong>Supported Diseases:</strong> ${data.components.treatment_service?.supported_diseases?.length || 0}</div>
                `;
            }
        } else {
            throw new Error('API not ready');
        }
    } catch (error) {
        elements.apiStatus.classList.remove('online');
        elements.apiStatus.classList.add('offline');
        elements.apiStatus.querySelector('.status-text').textContent = 'API Offline';

        // Retry in 5 seconds
        setTimeout(checkAPIHealth, 5000);
    }
}

// ============================================
// Toast Notifications
// ============================================
function showToast(type, title, message) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        success: '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
        error: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
        warning: '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
        info: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    };

    toast.innerHTML = `
        <svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            ${icons[type] || icons.info}
        </svg>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
        </button>
    `;

    elements.toastContainer.appendChild(toast);

    // Close button
    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.remove();
    });

    // Auto-remove after 5 seconds
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}
