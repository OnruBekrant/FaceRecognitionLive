/**
 * Face Recognition System - Main JavaScript
 * WebRTC-based camera implementation
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const videoContainer = document.querySelector('.video-container');
    const connectionStatus = document.getElementById('connectionStatus');
    const detectionList = document.getElementById('detectionList');
    const thresholdSlider = document.getElementById('detectionThreshold');
    const enableNotifications = document.getElementById('enableNotifications');
    const addPersonForm = document.getElementById('addPersonForm');
    const personNameInput = document.getElementById('personName');
    const captureFaceBtn = document.getElementById('captureFaceBtn');
    const savePersonBtn = document.getElementById('savePersonBtn');
    const capturePreview = document.getElementById('capturePreview');
    
    // State variables
    let isCaptureReady = false;
    let recentDetections = [];
    let streamingEnabled = true;
    let videoStream = null;
    let liveVideo = null;
    let canvas = null;
    let detectionOverlay = document.getElementById('detectionOverlay');
    let captureCanvas = null;
    let captureContext = null;
    let processingInterval = null;
    
    // Initialize the live video feed with WebRTC
    function initializeWebcam() {
        // Create video element to replace the image
        liveVideo = document.createElement('video');
        liveVideo.className = 'video-feed rounded';
        liveVideo.autoplay = true;
        liveVideo.width = 640;
        liveVideo.height = 480;
        
        // Create a canvas element for capturing frames
        canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        canvas.style.display = 'none';
        
        // Create a canvas for the capture preview in the modal
        captureCanvas = document.createElement('canvas');
        captureCanvas.width = 320;
        captureCanvas.height = 240;
        captureCanvas.style.display = 'none';
        
        // Replace the image source with the video element
        videoContainer.querySelector('img').replaceWith(liveVideo);
        
        // Add the canvases to the page
        document.body.appendChild(canvas);
        document.body.appendChild(captureCanvas);
        
        // Get access to the webcam
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then(function(stream) {
                // Set the video source to the webcam stream
                videoStream = stream;
                liveVideo.srcObject = stream;
                connectionStatus.textContent = 'Connected';
                connectionStatus.classList.replace('bg-danger', 'bg-success');
                showAlert('Webcam connected successfully!', 'success');
                
                // Start sending frames to the server for processing
                startFrameProcessing();
            })
            .catch(function(error) {
                console.error('Error accessing webcam:', error);
                connectionStatus.textContent = 'Disconnected';
                connectionStatus.classList.replace('bg-success', 'bg-danger');
                showAlert('Could not access webcam. Check permissions.', 'danger');
                
                // Fallback to the server-side simulation
                fallbackToServerSimulation();
            });
    }
    
    // Fallback to server-side simulation if webcam access fails
    function fallbackToServerSimulation() {
        const simulatedImg = document.createElement('img');
        simulatedImg.src = '/video_feed';
        simulatedImg.className = 'video-feed rounded';
        simulatedImg.width = 640;
        simulatedImg.height = 480;
        
        // Replace the video element with the image
        if (liveVideo) {
            liveVideo.replaceWith(simulatedImg);
        } else {
            videoContainer.querySelector('img').src = '/video_feed';
        }
        
        streamingEnabled = false;
        showAlert('Using simulated camera feed', 'info');
    }
    
    // Capture the current frame and send it to the server for processing
    function captureAndSendFrame() {
        if (!streamingEnabled || !liveVideo || !liveVideo.videoWidth) return;
        
        const context = canvas.getContext('2d');
        context.drawImage(liveVideo, 0, 0, canvas.width, canvas.height);
        
        // Convert the canvas to a data URL and send to the server
        canvas.toBlob(function(blob) {
            const formData = new FormData();
            formData.append('frame', blob, 'frame.jpg');
            
            fetch('/process_frame', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    updateDetectionOverlay(data.detections);
                    
                    // Add new detections to the list
                    if (data.detections && data.detections.length > 0) {
                        data.detections.forEach(detection => {
                            addDetectionToList({
                                name: detection.name,
                                similarity: detection.similarity,
                                isNew: false
                            });
                        });
                    }
                }
            })
            .catch(error => {
                console.error('Error sending frame to server:', error);
            });
        }, 'image/jpeg', 0.8);
    }
    
    // Update the detection overlay with face boxes and labels
    function updateDetectionOverlay(detections) {
        if (!detectionOverlay) return;
        
        // Clear the overlay
        detectionOverlay.innerHTML = '';
        
        // Add face boxes and labels for each detection
        if (detections && detections.length > 0) {
            detections.forEach(detection => {
                const loc = detection.location;
                
                // Create face box
                const faceBox = document.createElement('div');
                faceBox.className = 'face-box';
                faceBox.style.left = `${loc.left}px`;
                faceBox.style.top = `${loc.top}px`;
                faceBox.style.width = `${loc.right - loc.left}px`;
                faceBox.style.height = `${loc.bottom - loc.top}px`;
                
                // Create face label
                const faceLabel = document.createElement('div');
                faceLabel.className = 'face-label';
                faceLabel.style.left = `${loc.left}px`;
                faceLabel.style.top = `${loc.bottom + 5}px`;
                
                if (detection.name === 'Unknown') {
                    faceLabel.textContent = 'Unknown';
                } else {
                    faceLabel.textContent = `${detection.name} (${detection.similarity.toFixed(1)}%)`;
                }
                
                // Add to overlay
                detectionOverlay.appendChild(faceBox);
                detectionOverlay.appendChild(faceLabel);
            });
        }
    }
    
    // Start sending frames to the server at regular intervals
    function startFrameProcessing() {
        // Process frames every 500ms to avoid overwhelming the server
        processingInterval = setInterval(captureAndSendFrame, 500);
    }
    
    // Handle face capture for adding a new person
    captureFaceBtn.addEventListener('click', function() {
        if (personNameInput.value.trim() === '') {
            showAlert('Please enter a name for the person', 'warning');
            return;
        }
        
        // Capture the current frame for preview
        if (streamingEnabled && liveVideo && liveVideo.videoWidth) {
            // Draw the current frame on the capture canvas
            captureContext = captureCanvas.getContext('2d');
            captureContext.drawImage(liveVideo, 0, 0, captureCanvas.width, captureCanvas.height);
            
            // Show the preview
            const previewImg = document.createElement('img');
            previewImg.src = captureCanvas.toDataURL('image/jpeg');
            previewImg.className = 'rounded';
            previewImg.style.width = '100%';
            
            // Replace any existing preview
            while (capturePreview.firstChild) {
                capturePreview.removeChild(capturePreview.firstChild);
            }
            capturePreview.appendChild(previewImg);
        }
        
        // Enable save button to indicate capture is ready
        isCaptureReady = true;
        savePersonBtn.disabled = false;
        showAlert('Face captured! Click Save to add this person.', 'success');
    });
    
    // Handle save person button
    savePersonBtn.addEventListener('click', function() {
        if (!isCaptureReady) {
            showAlert('Please capture the face first', 'warning');
            return;
        }
        
        const personName = personNameInput.value.trim();
        
        // If we have a webcam stream, use the captured frame
        if (streamingEnabled && captureCanvas) {
            const imageData = captureCanvas.toDataURL('image/jpeg').split(',')[1];
            
            fetch('/add_person_webcam', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name: personName,
                    image_data: imageData
                })
            })
            .then(response => response.json())
            .then(handleAddPersonResponse)
            .catch(error => {
                console.error('Error adding person:', error);
                showAlert('Error adding person. Please try again.', 'danger');
            });
        } 
        // Otherwise use the server-side method
        else {
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
            .then(handleAddPersonResponse)
            .catch(error => {
                console.error('Error adding person:', error);
                showAlert('Error adding person. Please try again.', 'danger');
            });
        }
    });
    
    // Handle the response from adding a person
    function handleAddPersonResponse(data) {
        const personName = personNameInput.value.trim();
        
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
    }
    
    /**
     * Load known faces from the server
     */
    function loadKnownFaces() {
        fetch('/get_faces')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.faces.length > 0) {
                    console.log('Known faces loaded:', data.faces);
                    // Update the UI with the list of known people
                    updateKnownFacesUI(data.faces);
                }
            })
            .catch(error => {
                console.error('Error loading known faces:', error);
            });
    }
    
    // Update the UI to show known faces
    function updateKnownFacesUI(faces) {
        // You could add a panel or dropdown showing known people
        console.log('Known faces updated in UI');
    }
    
    /**
     * Add a detection to the recent detections list
     * @param {Object} detection - The detection data
     * @param {string} detection.name - The detected person's name
     * @param {number} detection.similarity - The similarity percentage
     * @param {boolean} detection.isNew - Whether this is a new person
     */
    function addDetectionToList(detection) {
        // Check if this person is already in recent detections
        const existingIndex = recentDetections.findIndex(d => d.name === detection.name);
        
        // If it's a new detection or an update with higher similarity
        if (existingIndex === -1 || recentDetections[existingIndex].similarity < detection.similarity) {
            // If it exists, remove it so we can add the updated version at the top
            if (existingIndex !== -1) {
                recentDetections.splice(existingIndex, 1);
            }
            
            // Add to recent detections array
            recentDetections.unshift(detection);
            
            // Keep only the last 5 detections
            if (recentDetections.length > 5) {
                recentDetections.pop();
            }
            
            // Update the UI
            updateDetectionsList();
            
            // Show notification if enabled and it's a known person
            if (enableNotifications.checked && detection.name !== 'Unknown') {
                showNotification(detection);
            }
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
            icon: '/static/images/icon.png' // The app icon
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
    
    // Initialize the webpage
    function init() {
        // Initialize webcam
        initializeWebcam();
        
        // Load known faces
        loadKnownFaces();
        
        // Reset capture state when modal is closed
        const addPersonModal = document.getElementById('addPersonModal');
        addPersonModal.addEventListener('hidden.bs.modal', function() {
            isCaptureReady = false;
            savePersonBtn.disabled = true;
            personNameInput.value = '';
            
            // Clear the capture preview
            while (capturePreview.firstChild) {
                capturePreview.removeChild(capturePreview.firstChild);
            }
        });
        
        // Update threshold value display when slider changes
        thresholdSlider.addEventListener('input', function() {
            document.getElementById('thresholdValue').textContent = `${this.value}%`;
            
            // Update the threshold on the server
            fetch('/update_threshold', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    threshold: parseInt(this.value)
                })
            }).catch(error => {
                console.error('Error updating threshold:', error);
            });
        });
    }
    
    // Start initialization
    init();
    
    // Clean up when the page is unloaded
    window.addEventListener('beforeunload', function() {
        if (processingInterval) {
            clearInterval(processingInterval);
        }
        
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
        }
    });
});
