/**
 * GeoVision WebSocket Client
 * Connects to the Flask server's WebSocket endpoint for real-time pipeline progress.
 */
(function() {
    'use strict';

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    let ws = null;
    let currentJobId = null;
    let eventListeners = {};
    let pipelinePhases = {};
    let results = null;
    let statusInterval = null;

    // -----------------------------------------------------------------------
    // WebSocket connection
    // -----------------------------------------------------------------------
    function connect(url) {
        url = url || `ws://${location.host}`;
        return new Promise((resolve, reject) => {
            ws = new WebSocket(url);

            ws.onopen = function() {
                console.log('[GeoVision] WebSocket connected');
                resolve(ws);
            };

            ws.onclose = function() {
                console.warn('[GeoVision] WebSocket disconnected');
                ws = null;
                emitEvent('ws:disconnected', {});
            };

            ws.onerror = function(err) {
                console.error('[GeoVision] WebSocket error:', err);
                reject(err);
            };

            ws.onmessage = function(e) {
                try {
                    const msg = JSON.parse(e.data);
                    handleWsMessage(msg);
                } catch (err) {
                    console.warn('[GeoVision] Non-JSON message:', e.data);
                }
            };
        });
    }

    function handleWsMessage(msg) {
        // Route by type
        switch (msg.type) {
            case 'phase:start':
                pipelinePhases[msg.phase] = { status: 'running', progress: 0, started_at: msg.time };
                emitEvent('phase:start', { phase: msg.phase, data: msg });
                updatePhaseUI();
                break;

            case 'phase:progress':
                if (pipelinePhases[msg.phase]) {
                    pipelinePhases[msg.phase].progress = msg.progress;
                    pipelinePhases[msg.phase].message = msg.message || '';
                }
                emitEvent('phase:progress', { phase: msg.phase, data: msg });
                updatePhaseUI();
                break;

            case 'phase:complete':
                if (pipelinePhases[msg.phase]) {
                    pipelinePhases[msg.phase].status = 'completed';
                    pipelinePhases[msg.phase].finished_at = msg.time;
                    pipelinePhases[msg.phase].result_summary = msg.summary || '';
                }
                emitEvent('phase:complete', { phase: msg.phase, data: msg });
                updatePhaseUI();
                break;

            case 'phase:failed':
                if (pipelinePhases[msg.phase]) {
                    pipelinePhases[msg.phase].status = 'failed';
                    pipelinePhases[msg.phase].error = msg.error || 'Unknown';
                }
                emitEvent('phase:failed', { phase: msg.phase, data: msg });
                updatePhaseUI();
                break;

            case 'log':
                emitEvent('log', { data: msg.data });
                break;

            case 'result':
                results = msg.data;
                emitEvent('result', { data: msg.data });
                break;

            case 'job:started':
                currentJobId = msg.job_id;
                emitEvent('job:started', { job_id: msg.job_id });
                break;

            case 'job:done':
                emitEvent('job:done', {});
                break;

            case 'error':
                emitEvent('error', { data: msg.data });
                break;

            case 'progress:overall':
                emitEvent('progress:overall', { progress: msg.progress, phase: msg.phase, message: msg.message });
                break;

            default:
                emitEvent('ws:message', { data: msg });
        }
    }

    // -----------------------------------------------------------------------
    // Event system
    // -----------------------------------------------------------------------
    function on(event, callback) {
        if (!eventListeners[event]) eventListeners[event] = [];
        eventListeners[event].push(callback);
    }

    function emitEvent(event, data) {
        const handlers = eventListeners[event] || [];
        handlers.forEach(cb => {
            try { cb(data); } catch (e) { console.error(`[${event}] handler error:`, e); }
        });
    }

    // -----------------------------------------------------------------------
    // Actions
    // -----------------------------------------------------------------------
    function send(data) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(data));
        }
    }

    // -----------------------------------------------------------------------
    // UI Helpers
    // -----------------------------------------------------------------------
    function updatePhaseUI() {
        const container = document.getElementById('phaseTracker');
        if (!container) return;

        container.innerHTML = '';
        const phaseOrder = [
            'phase0_exif_extraction',
            'phase0b_deepfake_detection',
            'phase1b_shadow_analysis',
            'phase1_visual_features',
            'phase2_ocr',
            'phase2b_vehicle_id',
            'phase3_deep_features',
            'phase3b_reverse_image_search',
            'phase3c_visual_geo',
            'phase3d_trained_models',
            'phase4_db_matching',
            'phase4b_property_records',
            'phase4c_chain_stores',
            'phase5_satellite_matching',
            'phase6_park_proximity',
            'phase7_cross_verification',
            'phase8b_vlm_reasoning',
            'phase9_synthesis',
        ];

        phaseOrder.forEach(phase => {
            const info = pipelinePhases[phase];
            if (!info) return;

            const labels = {
                'phase0_exif_extraction': 'EXIF Extraction',
                'phase0b_deepfake_detection': 'Deepfake Detection',
                'phase1b_shadow_analysis': 'Shadow/Solar Analysis',
                'phase1_visual_features': 'Visual Features',
                'phase2_ocr': 'OCR Text Extraction',
                'phase2b_vehicle_id': 'Vehicle Identification',
                'phase3_deep_features': 'Deep Features',
                'phase3b_reverse_image_search': 'Reverse Image Search',
                'phase3c_visual_geo': 'Visual Geo Engine',
                'phase3d_trained_models': 'Trained Models (GeoCLIP)',
                'phase4_db_matching': 'OSINT Data Matching',
                'phase4b_property_records': 'Property Records',
                'phase4c_chain_stores': 'Chain Store Locator',
                'phase5_satellite_matching': 'Satellite Matching',
                'phase6_park_proximity': 'Park Proximity',
                'phase7_cross_verification': 'Cross-View Verification',
                'phase8b_vlm_reasoning': 'VLM Reasoning',
                'phase9_synthesis': 'Synthesis & Ranking',
            };

            const icon = {
                'running': '<span class="phase-icon phase-running">&#9654;</span>',
                'completed': '<span class="phase-icon phase-complete">&#10003;</span>',
                'failed': '<span class="phase-icon phase-failed">&#10007;</span>',
            };

            const statusColor = {
                'running': '#f0ad4e',
                'completed': '#5cb85c',
                'failed': '#d9534f',
            };

            const row = document.createElement('div');
            row.className = 'phase-row';
            row.innerHTML = `
                <span class="phase-label">${labels[phase] || phase}</span>
                <span style="color:${statusColor[info.status]}">${icon[info.status]}</span>
                ${info.progress ? `<span class="phase-progress">${Math.round(info.progress)}%</span>` : ''}
                ${info.message ? `<span class="phase-message">${info.message}</span>` : ''}
            `;
            container.appendChild(row);
        });
    }

    // -----------------------------------------------------------------------
    // Results rendering
    // -----------------------------------------------------------------------
    function renderResults(data) {
        const container = document.getElementById('resultsArea');
        if (!container) return;

        const best = data.best_estimate || {};
        const estimates = data.location_estimates || [];
        const uncertainty = data.uncertainty || {};
        const execution = data.execution_summary || {};

        container.innerHTML = `
            <div class="results-header">
                <h2>Scan Complete</h2>
            </div>
            ${best.latitude && best.longitude ? `
            <div class="best-estimate">
                <div class="best-coords">${best.latitude.toFixed(6)}, ${best.longitude.toFixed(6)}</div>
                <div class="best-confidence">Confidence: ${(best.confidence * 100).toFixed(1)}%</div>
                <div class="best-sources">${(best.sources || []).join(', ')}</div>
            </div>
            ` : `
            <div class="no-results">
                <p>Could not determine a confident location for this image.</p>
                <p class="dim">Try an image with more visual cues: text, landmarks, road signs, or distinctive infrastructure.</p>
            </div>
            `}
            ${estimates.length > 1 ? `
            <div class="estimates-section">
                <h3>Ranked Candidates</h3>
                <table class="estimates-table">
                    <thead>
                        <tr>
                            <th>#</th><th>Latitude</th><th>Longitude</th><th>Confidence</th><th>Phase</th><th>Sources</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${estimates.map((e, i) => `
                            <tr>
                                <td>${i + 1}</td>
                                <td>${e.latitude.toFixed(6)}</td>
                                <td>${e.longitude.toFixed(6)}</td>
                                <td>${(e.confidence * 100).toFixed(1)}%</td>
                                <td>${e.phase || 'N/A'}</td>
                                <td><small>${(e.sources || []).join(', ')}</small></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            ` : ''}
            ${uncertainty.uncertainty_radius_km ? `
            <div class="uncertainty-section">
                <p><strong>Uncertainty:</strong> ±${uncertainty.uncertainty_radius_km} km (${uncertainty.granularity})</p>
            </div>
            ` : ''}
            <div class="execution-summary">
                <p>Phases: ${execution.phases_completed?.length || 0}/${execution.total_phases || 0} completed</p>
                ${(execution.phases_failed?.length || 0) > 0 ? `<p class="warn">${execution.phases_failed.length} phase(s) failed</p>` : ''}
            </div>
        `;
    }

    // -----------------------------------------------------------------------
    // Exports
    // -----------------------------------------------------------------------
    window.GeoVisionClient = {
        connect,
        on,
        send,
        updatePhaseUI,
        renderResults,
        getResults: () => results,
        getPhases: () => pipelinePhases,
        getJobId: () => currentJobId,
        disconnect: () => { if (ws) { ws.close(); ws = null; } }
    };

})();
