document.addEventListener('DOMContentLoaded', () => {
    const queryInput = document.querySelector('input[name="query"]');
    const voiceButton = document.getElementById('voice-button');
    const voiceStatus = document.getElementById('voice-status');

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        voiceButton.disabled = true;
        voiceStatus.textContent = 'Voice input not supported in this browser.';
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.addEventListener('start', () => {
        voiceStatus.textContent = 'Listening...';
        voiceButton.textContent = 'Stop';
        voiceButton.classList.add('listening');
    });

    recognition.addEventListener('end', () => {
        voiceStatus.textContent = 'Click the microphone to speak again.';
        voiceButton.textContent = '🎤 Voice input';
        voiceButton.classList.remove('listening');
    });

    recognition.addEventListener('result', (event) => {
        const transcript = event.results[0][0].transcript;
        queryInput.value = transcript;
        voiceStatus.textContent = `Captured: "${transcript}"`;
        // Auto-submit the form
        queryInput.form.submit();
    });

    recognition.addEventListener('error', (event) => {
        voiceStatus.textContent = `Voice error: ${event.error}`;
        voiceButton.textContent = '🎤 Voice input';
        voiceButton.classList.remove('listening');
    });

    voiceButton.addEventListener('click', () => {
        if (voiceButton.classList.contains('listening')) {
            recognition.stop();
        } else {
            voiceStatus.textContent = '';
            recognition.start();
        }
    });
});
