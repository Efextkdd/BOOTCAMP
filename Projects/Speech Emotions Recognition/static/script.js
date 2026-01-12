let hideTimer = null;

document.getElementById('predictBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('audioFile');
    const result = document.getElementById('result');

    if (!fileInput.files.length) {
        alert("Select a WAV file");
        return;
    }

    // Clear any previous timer
    if (hideTimer) clearTimeout(hideTimer);

    result.style.display = "block";
    result.innerText = "Predicting...";

    const formData = new FormData();
    formData.append("audio", fileInput.files[0]);

    try {
        const res = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        result.innerText = `Predicted Emotion: ${data.emotion}`;

        // stay for 10 seconds
        setTimeout(() => {
            result.style.display = "none";
        }, 10000);

    } catch (err) {
        console.error(err);
        result.innerText = "Prediction failed";}});