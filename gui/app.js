// Setup common chart configuration
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.9)';
Chart.defaults.plugins.tooltip.titleColor = '#f8fafc';
Chart.defaults.plugins.tooltip.bodyColor = '#f8fafc';
Chart.defaults.plugins.tooltip.borderColor = 'rgba(255,255,255,0.1)';
Chart.defaults.plugins.tooltip.borderWidth = 1;

const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
        duration: 0 // Turn off animation for smoother realtime updates
    },
    scales: {
        x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { maxTicksLimit: 10 }
        },
        y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
    },
    plugins: {
        legend: { labels: { usePointStyle: true, boxWidth: 8 } }
    },
    elements: {
        point: { radius: 0, hitRadius: 10, hoverRadius: 4 },
        line: { tension: 0.4, borderWidth: 2 }
    }
};

// Initialize Charts
let pressureChart, motorChart, tempChart, valveChart;

function initCharts() {
    const ctxPressure = document.getElementById('pressureChart').getContext('2d');
    pressureChart = new Chart(ctxPressure, {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'TP2 (bar)', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', fill: true },
            { label: 'TP3 (bar)', data: [], borderColor: '#8b5cf6', backgroundColor: 'rgba(139, 92, 246, 0.1)', fill: true }
        ]},
        options: { ...commonOptions, scales: { y: { ...commonOptions.scales.y, min: 6, max: 12 } } }
    });

    const ctxMotor = document.getElementById('motorChart').getContext('2d');
    motorChart = new Chart(ctxMotor, {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Motor Current (A)', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.1)', fill: true }
        ]},
        options: { ...commonOptions, scales: { y: { ...commonOptions.scales.y, min: 0, max: 10 } } }
    });

    const ctxTemp = document.getElementById('tempChart').getContext('2d');
    tempChart = new Chart(ctxTemp, {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Oil Temp (°C)', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', fill: true },
            { label: 'Reservoirs (bar)', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', fill: true }
        ]},
        options: commonOptions
    });

    const ctxValve = document.getElementById('valveChart').getContext('2d');
    valveChart = new Chart(ctxValve, {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'DV Pressure (bar)', data: [], borderColor: '#ec4899', yAxisID: 'y' },
            { label: 'H1 Valve', data: [], borderColor: '#06b6d4', borderDash: [5, 5], stepped: true, yAxisID: 'y1' }
        ]},
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: { type: 'linear', display: true, position: 'left' },
                y1: { type: 'linear', display: true, position: 'right', min: -0.5, max: 1.5, ticks: { stepSize: 1 } }
            }
        }
    });
}

// Format time
function formatTime(timestampStr) {
    if (!timestampStr) return '';
    try {
        const date = new Date(timestampStr);
        return date.toLocaleTimeString([], { hour12: false });
    } catch {
        return timestampStr.split(' ')[1] || timestampStr;
    }
}

// Update UI
function updateUI(data) {
    if (!data || data.length === 0) return;

    // 1. Update latest status and KPI
    const latest = data[data.length - 1];
    
    // Status
    const isAnomaly = latest.target === 1 || latest.status === 'Anomaly';
    const statusPanel = document.getElementById('system-status-panel');
    const statusValue = document.getElementById('system-status-value');
    
    if (isAnomaly) {
        statusPanel.className = 'status-panel anomaly';
        statusValue.textContent = 'ANOMALY DETECTED';
        document.getElementById('connection-status').style.backgroundColor = 'var(--status-anomaly)';
        document.getElementById('connection-status').style.boxShadow = '0 0 10px var(--status-anomaly)';
    } else {
        statusPanel.className = 'status-panel normal';
        statusValue.textContent = 'SYSTEM NORMAL';
        document.getElementById('connection-status').style.backgroundColor = 'var(--status-normal)';
        document.getElementById('connection-status').style.boxShadow = '0 0 10px var(--status-normal)';
    }

    // LSTM Anomaly Score
    if (latest.anomaly_score !== undefined && latest.anomaly_score !== null) {
        const score = parseFloat(latest.anomaly_score);
        document.getElementById('confidence-value').textContent = score.toFixed(1) + '%';
        const barFill = document.getElementById('confidence-bar');
        barFill.style.width = score + '%';
        
        if (score < 50) {
            barFill.style.background = 'var(--status-normal)';
            document.getElementById('confidence-value').style.color = 'var(--status-normal)';
        } else {
            barFill.style.background = 'var(--status-anomaly)';
            document.getElementById('confidence-value').style.color = 'var(--status-anomaly)';
        }
    } else {
        const elValue = document.getElementById('confidence-value');
        if(elValue) elValue.textContent = 'N/A';
        const elBar = document.getElementById('confidence-bar');
        if(elBar) elBar.style.width = '0%';
    }

    // KPIs
    document.getElementById('kpi-time').textContent = formatTime(latest.timestamp);
    document.getElementById('kpi-tp2').textContent = parseFloat(latest.TP2).toFixed(2) + ' bar';
    document.getElementById('kpi-tp3').textContent = parseFloat(latest.TP3).toFixed(2) + ' bar';
    document.getElementById('kpi-motor').textContent = parseFloat(latest.Motor_current).toFixed(2) + ' A';

    // 2. Update Charts
    const labels = data.map(d => formatTime(d.timestamp));
    
    // Pressure
    pressureChart.data.labels = labels;
    pressureChart.data.datasets[0].data = data.map(d => d.TP2);
    pressureChart.data.datasets[1].data = data.map(d => d.TP3);
    pressureChart.update();

    // Motor
    motorChart.data.labels = labels;
    motorChart.data.datasets[0].data = data.map(d => d.Motor_current);
    motorChart.update();

    // Temp & Reservoir
    tempChart.data.labels = labels;
    tempChart.data.datasets[0].data = data.map(d => d.Oil_temperature);
    tempChart.data.datasets[1].data = data.map(d => d.Reservoirs);
    tempChart.update();

    // Valve & DV
    valveChart.data.labels = labels;
    valveChart.data.datasets[0].data = data.map(d => d.DV_pressure);
    valveChart.data.datasets[1].data = data.map(d => d.H1);
    valveChart.update();
}

// Fetch Data
async function fetchData() {
    try {
        const response = await fetch('/api/data?limit=60');
        if (response.ok) {
            const result = await response.json();
            if (result.data && result.data.length > 0) {
                updateUI(result.data);
            }
        }
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

// Startup
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    fetchData(); // Initial fetch
    setInterval(fetchData, 2000); // Poll every 2 seconds
});
