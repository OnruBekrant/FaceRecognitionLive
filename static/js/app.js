/**
 * Face Recognition System - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const videoFeed = document.querySelector('.video-feed');
    const connectionStatus = document.getElementById('connectionStatus');
    const detectionList = document.getElementById('detectionList');
    const thresholdSlider = document.getElementById('detectionThreshold');
    const enableNotifications = document.getElementById('enableNotifications');
    const addPersonForm = document.getElementById('addPersonForm');
    const personNameInput = document.getElementById('personName');
    const captureFaceBtn = document.getElementById('captureFaceBtn');
    const savePersonBtn = document.getElementById('savePersonBtn');
    
    // State variables
    let isCaptureReady = false;
    let recentDetections = [];
    
    // Check if the video feed is working
    videoFeed.addEventListener('error', function() {
        connectionStatus.textContent = 'Disconnected';
        connectionStatus.classList.replace('bg-success', 'bg-danger');
        showAlert('Video feed error! Check your camera permissions.', 'danger');
    });
    
    // Make sure the connection status reflects the actual state
    videoFeed.addEventListener('load', function() {
        connectionStatus.textContent = 'Connected';
        connectionStatus.classList.replace('bg-danger', 'bg-success');
    });
    
    // Handle face capture for adding a new person
    captureFaceBtn.addEventListener('click', function() {
        if (personNameInput.value.trim() === '') {
            showAlert('Please enter a name for the person', 'warning');
            return;
        }
        
        // Enable save button to indicate capture is ready
        isCaptureReady = true;
        savePersonBtn.disabled = false;
        showAlert('Face capture ready! Click Save to add this person.', 'success');
    });
    
    // Handle save person button
    savePersonBtn.addEventListener('click', function() {
        if (!isCaptureReady) {
            showAlert('Please capture the face first', 'warning');
            return;
        }
        
        const personName = personNameInput.value.trim();
        
        // Send the person name to the server (current frame is captured on the server)
        fetch('/add_person', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: personName
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                showAlert(`Person "${personName}" added successfully!`, 'success');
                
                // Add the new person to the recent detections
                addDetectionToList({
                    name: personName,
                    similarity: 100,
                    isNew: true
                });
                
                // Reset form and close modal
                personNameInput.value = '';
                isCaptureReady = false;
                savePersonBtn.disabled = true;
                
                // Close the modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('addPersonModal'));
                modal.hide();
                
                // Load the known faces to update UI
                loadKnownFaces();
            } else {
                showAlert(`Error: ${data.message}`, 'danger');
            }
        })
        .catch(error => {
            console.error('Error adding person:', error);
            showAlert('Error adding person. Please try again.', 'danger');
        });
    });
    
    /**
     * Load known faces from the server
     */
    function loadKnownFaces() {
        fetch('/get_faces')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.faces.length > 0) {
                    console.log('Known faces loaded:', data.faces);
                    // Could update UI to show the list of known people
                }
            })
            .catch(error => {
                console.error('Error loading known faces:', error);
            });
    }
    
    /**
     * Add a detection to the recent detections list
     * @param {Object} detection - The detection data
     * @param {string} detection.name - The detected person's name
     * @param {number} detection.similarity - The similarity percentage
     * @param {boolean} detection.isNew - Whether this is a new person
     */
    function addDetectionToList(detection) {
        // Add to recent detections array
        recentDetections.unshift(detection);
        
        // Keep only the last 5 detections
        if (recentDetections.length > 5) {
            recentDetections.pop();
        }
        
        // Update the UI
        updateDetectionsList();
        
        // Show notification if enabled
        if (enableNotifications.checked && detection.name !== 'Unknown') {
            showNotification(detection);
        }
    }
    
    /**
     * Update the recent detections list in the UI
     */
    function updateDetectionsList() {
        // Clear the current list
        detectionList.innerHTML = '';
        
        if (recentDetections.length === 0) {
            const emptyItem = document.createElement('li');
            emptyItem.className = 'list-group-item text-center text-muted';
            emptyItem.textContent = 'No recent detections';
            detectionList.appendChild(emptyItem);
            return;
        }
        
        // Add each detection to the list
        recentDetections.forEach((detection, index) => {
            const item = document.createElement('li');
            item.className = 'list-group-item d-flex justify-content-between align-items-center';
            
            if (index === 0) {
                item.classList.add('highlight-detection');
            }
            
            const nameSpan = document.createElement('span');
            nameSpan.innerHTML = `<i class="fas fa-user me-2"></i>${detection.name}`;
            
            const badgeSpan = document.createElement('span');
            
            if (detection.name === 'Unknown') {
                badgeSpan.className = 'badge bg-secondary rounded-pill';
                badgeSpan.textContent = 'Unknown';
            } else {
                badgeSpan.className = 'badge bg-primary rounded-pill';
                badgeSpan.textContent = `${Math.round(detection.similarity)}% match`;
            }
            
            item.appendChild(nameSpan);
            item.appendChild(badgeSpan);
            detectionList.appendChild(item);
        });
    }
    
    /**
     * Show a notification for a detected person
     * @param {Object} detection - The detection data
     */
    function showNotification(detection) {
        // Check if browser supports notifications
        if (!("Notification" in window)) {
            console.log("This browser does not support notifications");
            return;
        }
        
        // Check if permission is already granted
        if (Notification.permission === "granted") {
            createNotification(detection);
        } 
        // Otherwise, request permission
        else if (Notification.permission !== "denied") {
            Notification.requestPermission().then(function (permission) {
                if (permission === "granted") {
                    createNotification(detection);
                }
            });
        }
    }
    
    /**
     * Create and show a notification
     * @param {Object} detection - The detection data
     */
    function createNotification(detection) {
        const options = {
            body: `Detected with ${Math.round(detection.similarity)}% similarity`,
            icon: '/static/icon.png' // This would be a static icon in a real app
        };
        
        const notification = new Notification(`Face Detected: ${detection.name}`, options);
        
        // Close the notification after 5 seconds
        setTimeout(notification.close.bind(notification), 5000);
    }
    
    /**
     * Show an alert message
     * @param {string} message - The message to show
     * @param {string} type - The alert type (success, danger, warning, info)
     */
    function showAlert(message, type) {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3`;
        alertDiv.style.zIndex = '1050';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Add to DOM
        document.body.appendChild(alertDiv);
        
        // Auto-dismiss after 3 seconds
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => {
                alertDiv.remove();
            }, 150);
        }, 3000);
    }
    
    // Initialize: Load known faces on page load
    loadKnownFaces();
    
    // Reset capture state when modal is closed
    const addPersonModal = document.getElementById('addPersonModal');
    addPersonModal.addEventListener('hidden.bs.modal', function() {
        isCaptureReady = false;
        savePersonBtn.disabled = true;
        personNameInput.value = '';
    });
    
    // Update threshold value display when slider changes
    thresholdSlider.addEventListener('input', function() {
        document.getElementById('thresholdValue').textContent = `${this.value}%`;
    });
});
