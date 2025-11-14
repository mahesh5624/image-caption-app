// Get DOM elements
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const changeBtn = document.getElementById('changeBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const error = document.getElementById('error');
const errorText = document.getElementById('errorText');
const tryAgainBtn = document.getElementById('tryAgainBtn');
const errorRetryBtn = document.getElementById('errorRetryBtn');

let selectedFile = null;

// Browse button click
browseBtn.addEventListener('click', () => {
    fileInput.click();
});

// Upload box click
uploadBox.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop handlers
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('drag-over');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('drag-over');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('drag-over');
    handleFileSelect(e.dataTransfer.files[0]);
});

// Handle file selection
function handleFileSelect(file) {
    if (!file) return;

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, or BMP).');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File is too large. Maximum size is 16MB.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadBox.style.display = 'none';
        previewSection.style.display = 'block';
        hideError();
        
        // Auto-generate caption after preview
        setTimeout(() => {
            generateCaption();
        }, 500);
    };
    reader.readAsDataURL(file);
}

// Generate caption
async function generateCaption() {
    if (!selectedFile) return;

    // Hide previous results
    results.style.display = 'none';
    error.style.display = 'none';
    loading.style.display = 'block';

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/generate-caption', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            displayResults(data.captions);
        } else {
            showError(data.error || 'Failed to generate caption. Please try again.');
        }
    } catch (err) {
        showError('Network error. Please check your connection and try again.');
        console.error('Error:', err);
    } finally {
        loading.style.display = 'none';
    }
}

// Display results
function displayResults(captions) {
    document.getElementById('captionGreedy').textContent = captions.greedy;
    document.getElementById('captionSamplingLow').textContent = captions.sampling_low;
    document.getElementById('captionSamplingHigh').textContent = captions.sampling_high;
    document.getElementById('captionNoRepeat').textContent = captions.no_repeat;
    
    results.style.display = 'block';
    
    // Scroll to results
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show error
function showError(message) {
    errorText.textContent = message;
    error.style.display = 'block';
    loading.style.display = 'none';
    results.style.display = 'none';
}

// Hide error
function hideError() {
    error.style.display = 'none';
}

// Change image button
changeBtn.addEventListener('click', () => {
    resetUpload();
});

// Try again buttons
tryAgainBtn.addEventListener('click', () => {
    resetUpload();
});

errorRetryBtn.addEventListener('click', () => {
    hideError();
    if (selectedFile) {
        generateCaption();
    } else {
        resetUpload();
    }
});

// Reset upload
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    uploadBox.style.display = 'block';
    results.style.display = 'none';
    loading.style.display = 'none';
    error.style.display = 'none';
    previewImage.src = '';
}