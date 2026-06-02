// Qunart Desktop App - Frontend Logic

// Wait for Eel to be ready
window.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const startBtn = document.getElementById('start-btn');
    const mainTitle = document.getElementById('main-status-title');
    const mainDesc = document.getElementById('main-status-desc');
    const hwBadge = document.getElementById('hw-badge');
    const terminalOutput = document.getElementById('terminal-output');
    
    // Progress
    const overallProgress = document.getElementById('overall-progress');
    const overallProgressText = document.getElementById('overall-progress-text');
    const stages = [
        document.getElementById('stage-1'),
        document.getElementById('stage-2'),
        document.getElementById('stage-3'),
        document.getElementById('stage-4'),
        document.getElementById('stage-5'),
        document.getElementById('stage-6')
    ];

    // Hardware polling
    setInterval(async () => {
        try {
            const stats = await eel.get_sys_stats()();
            
            document.querySelector('.cpu-fill').style.width = stats.cpu + '%';
            document.querySelector('.cpu-val').innerText = stats.cpu.toFixed(1) + '%';
            
            document.querySelector('.ram-fill').style.width = stats.ram + '%';
            document.querySelector('.ram-val').innerText = stats.ram.toFixed(1) + '%';
            
            document.querySelector('.gpu-fill').style.width = stats.gpu + '%';
            document.querySelector('.gpu-val').innerText = stats.gpu.toFixed(1) + '%';
        } catch(e) {
            console.error("Hardware polling error", e);
        }
    }, 1000);

    // Python -> JS Logging Callback
    eel.expose(ui_log);
    function ui_log(message) {
        const div = document.createElement('div');
        div.className = 'log-line';
        
        if (message.includes('❌')) {
            div.classList.add('error');
        } else if (message.includes('✅')) {
            div.classList.add('success');
        } else if (message.includes('⚠️')) {
            div.classList.add('warning');
        } else if (message.includes('📦') || message.includes('⚙️') || message.includes('🖥️')) {
            div.classList.add('info');
        }
        
        div.innerText = message;
        terminalOutput.appendChild(div);
        terminalOutput.scrollTop = terminalOutput.scrollHeight;
    }

    // Python -> JS Hardware Init
    eel.expose(ui_set_hardware);
    function ui_set_hardware(hw_name, has_cuda) {
        if (has_cuda) {
            hwBadge.innerHTML = `<span class="indicator green"></span> ${hw_name}`;
        } else {
            hwBadge.innerHTML = `<span class="indicator orange"></span> CPU Optimized`;
        }
    }

    // Python -> JS Stage Update
    eel.expose(ui_set_stage);
    function ui_set_stage(stage_index) {
        // Stage index 1 to 6
        stages.forEach((el, idx) => {
            if (idx + 1 < stage_index) {
                el.className = 'stage done';
            } else if (idx + 1 === stage_index) {
                el.className = 'stage active';
            } else {
                el.className = 'stage pending';
            }
        });

        // Update overall
        const pct = Math.round(((stage_index - 1) / 6) * 100);
        overallProgress.style.width = pct + '%';
        overallProgressText.innerText = pct + '%';
    }

    // Python -> JS Completion
    eel.expose(ui_set_complete);
    function ui_set_complete(metrics) {
        stages.forEach(el => el.className = 'stage done');
        overallProgress.style.width = '100%';
        overallProgressText.innerText = '100%';

        document.getElementById('val-orig-size').innerText = metrics.original_size_gb.toFixed(3) + ' GB';
        document.getElementById('val-comp-size').innerText = metrics.quantized_size_gb.toFixed(3) + ' GB';
        document.getElementById('val-ratio').innerText = metrics.compression_ratio.toFixed(2) + 'x';
        document.getElementById('val-quality').innerText = metrics.quality_retention.toFixed(1) + '%';

        mainTitle.innerText = "Compression Successful";
        mainDesc.innerText = "The model is optimized and saved to disk.";
        document.querySelector('.status-icon').className = 'status-icon success';
        document.querySelector('.status-icon i').className = 'fa-solid fa-check-double';
        
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Run Again';
    }

    // Python -> JS Error
    eel.expose(ui_set_error);
    function ui_set_error(msg) {
        mainTitle.innerText = "Pipeline Failed";
        mainDesc.innerText = msg;
        document.querySelector('.status-icon').className = 'status-icon error';
        document.querySelector('.status-icon').style.borderColor = 'var(--danger)';
        document.querySelector('.status-icon').style.color = 'var(--danger)';
        document.querySelector('.status-icon i').className = 'fa-solid fa-triangle-exclamation';
        
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Retry';
    }

    // Clear Logs
    document.getElementById('clear-log-btn').addEventListener('click', () => {
        terminalOutput.innerHTML = '<div class="log-line system">[SYSTEM] Console cleared.</div>';
    });

    // Start Button Click
    startBtn.addEventListener('click', async () => {
        startBtn.disabled = true;
        startBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Processing...';
        
        mainTitle.innerText = "Pipeline Active";
        mainDesc.innerText = "Executing quantum hybrid compression...";
        document.querySelector('.status-icon').className = 'status-icon processing';
        document.querySelector('.status-icon i').className = 'fa-solid fa-gear';
        
        // Reset UI
        document.getElementById('val-orig-size').innerText = '-- GB';
        document.getElementById('val-comp-size').innerText = '-- GB';
        document.getElementById('val-ratio').innerText = '-- x';
        document.getElementById('val-quality').innerText = '-- %';
        terminalOutput.innerHTML = '';
        ui_log('[SYSTEM] Initiating Qunart Pipeline...');

        // Start Python Backend Task
        await eel.start_pipeline()();
    });
    
    // Initial request to get hardware
    eel.get_initial_hardware()();
});
