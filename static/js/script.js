// Function to detect phishing and display result
function detectPhishing() {
    var domain = document.getElementById('domainInput').value.trim(); // Trim whitespace from the input
    if (!domain) {
        alert("Please enter a domain name.");
        return;
    }

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ domain: domain })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error); });
        }
        return response.json();
    })
    .then(data => {
        if (data.result !== undefined) {
            document.getElementById('predictionResult').innerText = 'Prediction: ' + data.result;
        } else if (data.error) {
            document.getElementById('predictionResult').innerText = 'Error: ' + data.error;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('predictionResult').innerText = 'Error: ' + error.message;
    });
}

// Function to navigate to a different page
function goToPage(page) {
    window.location.href = '/' + page;
}
