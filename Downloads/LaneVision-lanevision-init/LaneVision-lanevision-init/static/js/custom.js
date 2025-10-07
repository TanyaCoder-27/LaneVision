// Custom JavaScript for Vehicle Speed Detection
document.addEventListener('DOMContentLoaded', function() {
    
    // File upload validation and preview
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                validateFile(file);
            }
        });
    });
    
    function validateFile(file) {
        const maxSize = 500 * 1024 * 1024; // 500MB
        const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime'];
        
        if (file.size > maxSize) {
            showAlert('File size exceeds 500MB limit!', 'danger');
            return false;
        }
        
        if (!allowedTypes.includes(file.type)) {
            showAlert('Please select a valid video file (MP4, AVI, MOV)!', 'danger');
            return false;
        }
        
        showAlert('File selected successfully!', 'success');
        return true;
    }
    
    // Show loading overlay for processing
    const processButtons = document.querySelectorAll('[href*="process"]');
    processButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            if (confirm('Video processing may take several minutes. Continue?')) {
                showLoadingOverlay('Processing video... Please wait.');
            } else {
                e.preventDefault();
            }
        });
    });
    
    // Handle download buttons - no loading overlay
    // Removed loading overlay for direct downloads
    
    function showLoadingOverlay(message) {
        // Remove any existing overlay first
        removeLoadingOverlay();
        
        const overlay = document.createElement('div');
        overlay.className = 'loading-overla';
        overlay.innerHTML = `
            <div class="text-center" style="display:none;">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3">${message}</p>
            </div>
        `;
        document.body.appendChild(overlay);
        
        // Add show class after a small delay to trigger transition
        setTimeout(() => {
            overlay.classList.add('show');
        }, 10);
    }
    
    function removeLoadingOverlay() {
        const existingOverlay = document.querySelector('.loading-overla');
        if (existingOverlay) {
            existingOverlay.classList.remove('show');
            setTimeout(() => {
                existingOverlay.remove();
            }, 300);
        }
    }
    
    // Auto-dismiss alerts
    setTimeout(function() {
        const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
        alerts.forEach(alert => {
            if (alert.classList.contains('alert-success')) {
                setTimeout(() => {
                    alert.style.opacity = '0';
                    setTimeout(() => alert.remove(), 300);
                }, 3000);
            }
        });
    }, 100);
    
    // Confirm delete actions
    const deleteButtons = document.querySelectorAll('[onclick*="confirm"]');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            if (!confirm('Are you sure you want to delete this item? This action cannot be undone.')) {
                e.preventDefault();
            }
        });
    });
    
    // Table row highlighting
    const tableRows = document.querySelectorAll('table tbody tr');
    tableRows.forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.style.backgroundColor = 'rgba(13, 110, 253, 0.1)';
        });
        
        row.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '';
        });
    });
    
    // Video player controls
    const videoElements = document.querySelectorAll('video');
    videoElements.forEach(video => {
        // Add video-loading class initially
        video.classList.add('video-loading');
        
        video.addEventListener('loadstart', function() {
            console.log('Video loading started');
            this.classList.add('video-loading');
        });
        
        video.addEventListener('loadeddata', function() {
            console.log('Video data loaded');
            this.classList.remove('video-loading');
        });
        
        video.addEventListener('canplay', function() {
            console.log('Video can play');
            this.classList.remove('video-loading');
        });
        
        video.addEventListener('error', function(e) {
            console.error('Video error:', this.error);
            this.classList.remove('video-loading');
            showAlert('Error loading video. Please try the alternative player or download the video.', 'warning');
            document.getElementById('fallback-player').classList.remove('d-none');
        });
        
        // Add timeout to check if video is still loading after 5 seconds
        setTimeout(() => {
            if (video.readyState < 3) { // HAVE_FUTURE_DATA = 3
                console.warn('Video taking too long to load');
                document.getElementById('fallback-player').classList.remove('d-none');
            }
        }, 5000);
        
        video.addEventListener('error', function() {
            showAlert('Error loading video. Please try again.', 'danger');
        });
    });
    
});

// Utility function to show alerts
function showAlert(message, type = 'info') {
    const alertContainer = document.querySelector('.container');
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show mt-3`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    if (alertContainer) {
        alertContainer.insertBefore(alert, alertContainer.firstChild);
    }
    
    // Auto dismiss success alerts
    if (type === 'success') {
        setTimeout(() => {
            alert.remove();
        }, 5000);
    }
}

// Progress bar for file uploads
function updateProgress(percent) {
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        progressBar.style.width = percent + '%';
        progressBar.setAttribute('aria-valuenow', percent);
        progressBar.textContent = Math.round(percent) + '%';
    }
}
