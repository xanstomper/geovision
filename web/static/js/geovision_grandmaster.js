/**
 * GeoVision Grandmaster Control HUD
 * Zero-Storage Geolocation Suite, Physical Forensics, and Interactive GIS
 */
(function() {
    'use strict';

    // Global state
    let activeMode = 'grandmaster'; // 'grandmaster' | 'forensics' | 'deep'
    let selectedImageFile = null;
    let grandmasterResult = null;
    let leafletMap = null;
    let mapMarkers = [];
    let mapLayers = {};

    document.addEventListener('DOMContentLoaded', () => {
        initModeSwitcher();
        initDropZone();
        initControls();
    });

    // -----------------------------------------------------------------------
    // Mode Switcher
    // -----------------------------------------------------------------------
    function initModeSwitcher() {
        const buttons = document.querySelectorAll('.mode-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                buttons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                activeMode = btn.dataset.mode;
                updateModeUI();
            });
        });
    }

    function updateModeUI() {
        const hintContainer = document.getElementById('regionHintContainer');
        const startBtn = document.getElementById('startScan');
        const modeBadge = document.getElementById('activeModeBadge');

        if (modeBadge) {
            if (activeMode === 'grandmaster') {
                modeBadge.textContent = 'MODE: Grandmaster Zero-Storage';
                modeBadge.className = 'badge badge-grandmaster';
            } else if (activeMode === 'forensics') {
                modeBadge.textContent = 'MODE: Physical Forensics (<100ms)';
                modeBadge.className = 'badge badge-forensics';
            } else {
                modeBadge.textContent = 'MODE: Deep Multi-Phase Pipeline';
                modeBadge.className = 'badge badge-deep';
            }
        }

        if (activeMode === 'forensics') {
            if (hintContainer) hintContainer.style.display = 'none';
            if (startBtn) startBtn.innerHTML = '&#9889; Run Instant Forensics';
        } else if (activeMode === 'grandmaster') {
            if (hintContainer) hintContainer.style.display = '';
            if (startBtn) startBtn.innerHTML = '&#127919; Pinpoint Location (Zero-Storage)';
        } else {
            if (hintContainer) hintContainer.style.display = '';
            if (startBtn) startBtn.innerHTML = '&#9654; Start Deep Pipeline Scan';
        }
    }

    // -----------------------------------------------------------------------
    // Drag & Drop / File Selection
    // -----------------------------------------------------------------------
    function initDropZone() {
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const previewArea = document.getElementById('imagePreviewArea');

        if (!dropZone || !fileInput) return;

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                handleSelectedFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleSelectedFile(fileInput.files[0]);
            }
        });
    }

    function handleSelectedFile(file) {
        if (!file.type.startsWith('image/')) return;
        selectedImageFile = file;

        const previewArea = document.getElementById('imagePreviewArea');
        const startBtn = document.getElementById('startScan');
        const fileInfo = document.getElementById('fileInfoText');

        if (previewArea) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewArea.innerHTML = `
                    <div class="preview-card">
                        <img src="${e.target.result}" alt="Target Preview" class="img-preview" />
                        <div class="preview-meta">
                            <strong>${file.name}</strong> &bull; ${(file.size / 1024).toFixed(1)} KB
                        </div>
                    </div>
                `;
            };
            reader.readAsDataURL(file);
        }

        if (fileInfo) {
            fileInfo.textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
        }

        if (startBtn) {
            startBtn.disabled = false;
        }
    }

    // -----------------------------------------------------------------------
    // Execution & Controls
    // -----------------------------------------------------------------------
    function initControls() {
        const startBtn = document.getElementById('startScan');
        const clearBtn = document.getElementById('clearFiles');
        const newScanBtn = document.getElementById('newScan');

        if (startBtn) {
            startBtn.addEventListener('click', executeScan);
        }

        if (clearBtn) {
            clearBtn.addEventListener('click', resetUpload);
        }

        if (newScanBtn) {
            newScanBtn.addEventListener('click', () => {
                document.getElementById('resultsSection').style.display = 'none';
                document.getElementById('uploadSection').style.display = '';
            });
        }
    }

    function resetUpload() {
        selectedImageFile = null;
        const fileInput = document.getElementById('fileInput');
        if (fileInput) fileInput.value = '';
        const previewArea = document.getElementById('imagePreviewArea');
        if (previewArea) previewArea.innerHTML = '';
        const fileInfo = document.getElementById('fileInfoText');
        if (fileInfo) fileInfo.textContent = '';
        const startBtn = document.getElementById('startScan');
        if (startBtn) startBtn.disabled = true;
    }

    async function executeScan() {
        if (!selectedImageFile) return;

        if (activeMode === 'deep') {
            // Hand off to existing WebSocket / SSE flow in index.html
            if (window.runDeepScan) {
                window.runDeepScan(selectedImageFile);
            }
            return;
        }

        const uploadSection = document.getElementById('uploadSection');
        const progressSection = document.getElementById('progressSection');
        const resultsSection = document.getElementById('resultsSection');
        const progressPercent = document.getElementById('progressPercent');
        const logArea = document.getElementById('logArea');

        uploadSection.style.display = 'none';
        progressSection.style.display = '';
        resultsSection.style.display = 'none';
        if (logArea) logArea.innerHTML = '';

        const appendLog = (msg, level = 'info') => {
            if (!logArea) return;
            const line = document.createElement('div');
            line.className = level;
            line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            logArea.appendChild(line);
            logArea.scrollTop = logArea.scrollHeight;
        };

        const fd = new FormData();
        fd.append('image', selectedImageFile);

        const hintInput = document.getElementById('regionHint');
        const hint = hintInput ? hintInput.value.trim() : '';
        if (hint) fd.append('hint', hint);

        try {
            if (activeMode === 'forensics') {
                if (progressPercent) progressPercent.textContent = 'Extracting Physical Forensics...';
                appendLog('Running ultra-fast sub-100ms physical forensics...');

                const t0 = performance.now();
                const resp = await fetch('/api/forensics', { method: 'POST', body: fd });
                const res = await resp.json();
                const dt = (performance.now() - t0).toFixed(1);

                appendLog(`Forensics extracted in ${dt} ms. Rendering report.`);
                progressSection.style.display = 'none';
                resultsSection.style.display = '';
                renderForensicsResults(res.forensics || res);

            } else if (activeMode === 'grandmaster') {
                if (progressPercent) progressPercent.textContent = 'Running Grandmaster Zero-Storage Suite...';
                appendLog('Phase 1: Physical Visual Forensics & Ephemeris...', 'info');
                appendLog('Phase 2: PlonkIt 85-Country Elimination Lattice...', 'info');
                appendLog('Phase 3: CarID Silhouette & Fleet Demographics...', 'info');
                appendLog('Phase 4: Topological Micro-GIS Street Intersection Targeter...', 'info');
                appendLog('Phase 5: SkyEye Overhead Cross-View Satellite Footprint Verifier...', 'info');

                const t0 = performance.now();
                const resp = await fetch('/api/grandmaster', { method: 'POST', body: fd });
                const data = await resp.json();
                const dt = ((performance.now() - t0) / 1000).toFixed(2);

                grandmasterResult = data;
                appendLog(`Grandmaster investigation complete in ${dt}s! Pinned coordinates acquired.`, 'info');

                progressSection.style.display = 'none';
                resultsSection.style.display = '';
                renderGrandmasterResults(data);
            }
        } catch (err) {
            appendLog(`Execution error: ${err.message}`, 'error');
            if (progressPercent) progressPercent.textContent = 'Failed!';
        }
    }

    // -----------------------------------------------------------------------
    // Grandmaster HUD Rendering
    // -----------------------------------------------------------------------
    function renderGrandmasterResults(data) {
        const resultsArea = document.getElementById('resultsArea');
        if (!resultsArea) return;

        const best = data.best_estimate || {};
        const lat = best.latitude ? best.latitude.toFixed(6) : '0.000000';
        const lon = best.longitude ? best.longitude.toFixed(6) : '0.000000';
        const conf = best.confidence ? (best.confidence * 100).toFixed(1) : '50.0';
        const tier = (best.precision_tier || 'street_level').replace(/_/g, ' ').toUpperCase();
        const street = best.street_name || best.road_class || 'Target Corridor';
        const inter = best.intersection || 'Nearest Cross Street';
        const city = best.city || 'Detected Metropolitan Area';
        const country = best.country || 'Detected Region';

        const gmapsUrl = best.google_maps_url || `https://www.google.com/maps/search/?api=1&query=${lat},${lon}`;
        const osmUrl = best.osm_url || `https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=18/${lat}/${lon}`;
        const streetViewUrl = `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${lat},${lon}`;
        const esriUrl = `https://www.arcgis.com/home/webmap/viewer.html?center=${lon},${lat}&level=19`;

        const plonkit = data.plonkit_evaluation || {};
        const candidates = plonkit.candidates || [];
        const topCandidates = candidates.slice(0, 5);
        const eliminatedCount = candidates.filter(c => c.falsified).length;

        const carid = data.car_fleet || {};
        const demo = carid.demographics || {};
        const vehicleTypes = carid.vehicle_types || {};

        const forensics = data.forensic_breakdown || {};
        const solar = forensics.solar_shadow || {};
        const driveSide = forensics.driving_side ? forensics.driving_side.driving_side : 'unknown';
        const roadMark = forensics.road_markings ? forensics.road_markings.line_color : 'unknown';
        const pole = forensics.utility_pole ? forensics.utility_pole.dominant_type : 'standard';

        const reasoning = data.reasoning_chain || [];

        resultsArea.innerHTML = `
            <!-- Top Action Header -->
            <div class="grandmaster-topbar">
                <div class="badge-group">
                    <span class="hud-pill hud-pill-gold">&#127919; GRANDMASTER ZERO-STORAGE</span>
                    <span class="hud-pill hud-pill-cyan">${tier}</span>
                    <span class="hud-pill hud-pill-green">&#10003; 0 REFERENCE PHOTOS NEEDED</span>
                </div>
                <div class="dossier-actions">
                    <button class="btn btn-sm btn-outline" id="exportMdDossier">&#128196; Export Markdown Dossier</button>
                    <button class="btn btn-sm btn-outline" id="exportHtmlDossier">&#127760; Export HTML Dossier</button>
                </div>
            </div>

            <!-- Pinpoint Primary Verdict Card -->
            <div class="pinpoint-card">
                <div class="pinpoint-header">
                    <div>
                        <div class="pinpoint-title">&#128205; ${street}</div>
                        <div class="pinpoint-sub">${inter ? '&times; ' + inter + ' &bull; ' : ''}${city}, ${country}</div>
                    </div>
                    <div class="pinpoint-confidence">
                        <div class="conf-value">${conf}%</div>
                        <div class="conf-label">Confidence</div>
                    </div>
                </div>

                <div class="coords-box">
                    <span class="coords-text" id="coordsDisplay">${lat}, ${lon}</span>
                    <button class="btn-copy" id="btnCopyCoords" title="Copy coordinates">&#128203; Copy</button>
                </div>

                <div class="pinpoint-links">
                    <a href="${streetViewUrl}" target="_blank" class="hud-btn hud-btn-streetview">&#128694; Google Street View</a>
                    <a href="${osmUrl}" target="_blank" class="hud-btn hud-btn-osm">&#127758; OpenStreetMap</a>
                    <a href="${gmapsUrl}" target="_blank" class="hud-btn hud-btn-gmaps">&#128506; Google Maps</a>
                    <a href="${esriUrl}" target="_blank" class="hud-btn hud-btn-esri">&#128752; ESRI World Satellite</a>
                </div>
            </div>

            <!-- Intelligence Grid (2x2 Matrix) -->
            <div class="hud-grid">
                <!-- Card 1: PlonkIt 85-Country Lattice -->
                <div class="hud-card">
                    <div class="hud-card-header">
                        <span class="hud-card-title">&#127760; PlonkIt 85-Country Elimination Lattice</span>
                        <span class="hud-tag">${eliminatedCount} Countries Ruled Out</span>
                    </div>
                    <div class="hud-card-body">
                        <div class="tag-lattice">
                            <span class="lattice-tag"><strong>Drive Side:</strong> ${driveSide.toUpperCase()}</span>
                            <span class="lattice-tag"><strong>Road Lines:</strong> ${roadMark.toUpperCase()}</span>
                            <span class="lattice-tag"><strong>Utility Pole:</strong> ${pole.replace(/_/g, ' ')}</span>
                        </div>

                        <div style="margin-top: 14px;">
                            <div class="section-label">TOP MATCHING JURISDICTIONS</div>
                            ${topCandidates.map(c => `
                                <div class="candidate-row">
                                    <div class="candidate-name">
                                        <span class="candidate-iso">${c.iso}</span>
                                        <span>${c.country}</span>
                                    </div>
                                    <div class="candidate-bar-wrap">
                                        <div class="candidate-bar" style="width: ${Math.round(c.confidence * 100)}%;"></div>
                                    </div>
                                    <span class="candidate-pct">${Math.round(c.confidence * 100)}%</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>

                <!-- Card 2: CarID Fleet Demographics -->
                <div class="hud-card">
                    <div class="hud-card-header">
                        <span class="hud-card-title">&#128663; CarID Vehicle Fleet Demographics</span>
                        <span class="hud-tag">${carid.vehicles_detected || 0} Vehicles Identified</span>
                    </div>
                    <div class="hud-card-body">
                        <div class="fleet-demographics-grid">
                            <div class="demo-metric">
                                <div class="demo-val">${demo.pickup_suv_ratio !== undefined ? (demo.pickup_suv_ratio * 100).toFixed(0) + '%' : '0%'}</div>
                                <div class="demo-lbl">Pickup &amp; SUV Ratio</div>
                            </div>
                            <div class="demo-metric">
                                <div class="demo-val">${demo.kei_car_ratio !== undefined ? (demo.kei_car_ratio * 100).toFixed(0) + '%' : '0%'}</div>
                                <div class="demo-lbl">Kei Car Ratio</div>
                            </div>
                            <div class="demo-metric">
                                <div class="demo-val">${demo.sedan_hatchback_ratio !== undefined ? (demo.sedan_hatchback_ratio * 100).toFixed(0) + '%' : '0%'}</div>
                                <div class="demo-lbl">Sedan / Hatchback</div>
                            </div>
                        </div>

                        <div style="margin-top: 14px;">
                            <div class="section-label">DETECTED VEHICLE SILHOUETTES</div>
                            <div class="tag-lattice">
                                <span class="lattice-tag">Sedans: ${vehicleTypes.sedan || 0}</span>
                                <span class="lattice-tag">SUVs: ${vehicleTypes.suv || 0}</span>
                                <span class="lattice-tag">Pickups: ${vehicleTypes.pickup || 0}</span>
                                <span class="lattice-tag">Trucks/Buses: ${(vehicleTypes.truck || 0) + (vehicleTypes.bus || 0)}</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Card 3: Solar Lock Ephemeris & Latitude Band -->
                <div class="hud-card">
                    <div class="hud-card-header">
                        <span class="hud-card-title">&#9728;&#65039; Solar Lock Ephemeris &amp; Latitude Band</span>
                        <span class="hud-tag">${solar.inferred_hemisphere ? solar.inferred_hemisphere.toUpperCase() : 'HEMISPHERE BOUND'}</span>
                    </div>
                    <div class="hud-card-body">
                        <div class="solar-stats-grid">
                            <div class="solar-stat">
                                <div class="solar-val">${solar.shadow_ratio !== undefined ? solar.shadow_ratio.toFixed(2) : 'N/A'}</div>
                                <div class="solar-lbl">Shadow / Height Ratio</div>
                            </div>
                            <div class="solar-stat">
                                <div class="solar-val">${solar.solar_elevation_deg !== undefined ? solar.solar_elevation_deg.toFixed(1) + '&deg;' : 'N/A'}</div>
                                <div class="solar-lbl">Sun Elevation (&alpha;)</div>
                            </div>
                            <div class="solar-stat">
                                <div class="solar-val">${solar.latitude_bounds ? solar.latitude_bounds[0].toFixed(0) + '&deg; to ' + solar.latitude_bounds[1].toFixed(0) + '&deg;' : 'Global'}</div>
                                <div class="solar-lbl">Bounded Latitude Band</div>
                            </div>
                        </div>
                        <div class="solar-note" style="margin-top: 12px; font-size: 12px; color: var(--text-dim);">
                            Derived from solar declination equations &amp; OpenCV shadow angles. Eliminates invalid latitudinal zones worldwide.
                        </div>
                    </div>
                </div>

                <!-- Card 4: SkyEye Cross-View Satellite Footprint -->
                <div class="hud-card">
                    <div class="hud-card-header">
                        <span class="hud-card-title">&#128752; SkyEye Cross-View Footprint Verifier</span>
                        <span class="hud-tag">ESRI Orthophoto Cross-Match</span>
                    </div>
                    <div class="hud-card-body">
                        <p style="font-size: 13px; color: var(--text-dim); margin-bottom: 12px;">
                            Extracts ground building facade perspective yaw &amp; rooflines, cross-referencing vector building polygons from OpenStreetMap.
                        </p>
                        <button class="btn btn-primary btn-sm" id="btnRunSkyEye" data-lat="${lat}" data-lon="${lon}">
                            &#128752; Scan Building Footprints Around Pin
                        </button>
                        <div id="skyEyeResultsArea" style="margin-top: 12px; font-size: 12px;"></div>
                    </div>
                </div>
            </div>

            <!-- Reasoning Deduction Chain -->
            ${reasoning.length > 0 ? `
            <div class="reasoning-card">
                <div class="section-label">&#128269; ZERO-STORAGE MULTI-AGENT REASONING CHAIN</div>
                <ol class="reasoning-list">
                    ${reasoning.map(r => `<li>${r}</li>`).join('')}
                </ol>
            </div>
            ` : ''}
        `;

        // Wire Dossier Export buttons
        const mdBtn = document.getElementById('exportMdDossier');
        if (mdBtn) {
            mdBtn.addEventListener('click', () => exportDossier('markdown'));
        }
        const htmlBtn = document.getElementById('exportHtmlDossier');
        if (htmlBtn) {
            htmlBtn.addEventListener('click', () => exportDossier('html'));
        }

        // Wire Copy Coords button
        const copyBtn = document.getElementById('btnCopyCoords');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(`${lat}, ${lon}`);
                copyBtn.textContent = '&#10003; Copied!';
                setTimeout(() => { copyBtn.innerHTML = '&#128203; Copy'; }, 2000);
            });
        }

        // Wire SkyEye Interactive Button
        const skyEyeBtn = document.getElementById('btnRunSkyEye');
        if (skyEyeBtn) {
            skyEyeBtn.addEventListener('click', async () => {
                const targetLat = skyEyeBtn.dataset.lat;
                const targetLon = skyEyeBtn.dataset.lon;
                const outDiv = document.getElementById('skyEyeResultsArea');
                outDiv.innerHTML = '<span class="spinner"></span> Querying OpenStreetMap Overpass &amp; ESRI satellite polygons...';
                try {
                    const fd = new FormData();
                    fd.append('lat', targetLat);
                    fd.append('lon', targetLon);
                    fd.append('radius_m', '400');
                    if (selectedImageFile) fd.append('image', selectedImageFile);

                    const r = await fetch('/api/skyeye', { method: 'POST', body: fd });
                    const res = await r.json();
                    if (res.status === 'success') {
                        outDiv.innerHTML = `
                            <div style="background:#0d1117;padding:8px;border-radius:6px;border:1px solid var(--border);">
                                <strong style="color:var(--success);">&#10003; ${res.footprints_found} Building Polygons Corroborated!</strong><br>
                                Facade Roofline: <code>${res.facade_features?.roof_type || 'flat'}</code> &bull; Facade Yaw: <code>${res.facade_features?.facade_yaw_deg || 0}&deg;</code><br>
                                <a href="${res.satellite_ortho_url}" target="_blank" style="color:var(--accent);text-decoration:underline;">Inspect 1m High-Res ESRI Tile &rarr;</a>
                            </div>
                        `;
                    } else {
                        outDiv.innerHTML = `<span style="color:var(--danger)">Error: ${res.error || 'Failed'}</span>`;
                    }
                } catch (e) {
                    outDiv.innerHTML = `<span style="color:var(--danger)">Error: ${e.message}</span>`;
                }
            });
        }

        // Render Map with high-res tiles
        renderInteractiveMap(parseFloat(lat), parseFloat(lon), data);
    }

    // -----------------------------------------------------------------------
    // Forensics Only Mode Rendering (<100ms)
    // -----------------------------------------------------------------------
    function renderForensicsResults(forensics) {
        const resultsArea = document.getElementById('resultsArea');
        if (!resultsArea) return;

        const ds = forensics.driving_side || {};
        const rm = forensics.road_markings || {};
        const up = forensics.utility_pole || {};
        const lp = forensics.license_plate || {};
        const sb = forensics.soil_and_biome || {};
        const ss = forensics.solar_shadow || {};

        resultsArea.innerHTML = `
            <div class="grandmaster-topbar">
                <span class="hud-pill hud-pill-cyan">&#9889; PHYSICAL FORENSICS (&lt;100ms)</span>
                <span class="hud-pill hud-pill-gold">HARD GEOMETRIC &amp; SPECTRAL EVIDENCE</span>
            </div>

            <div class="hud-grid" style="margin-top: 16px;">
                <div class="hud-card">
                    <div class="hud-card-header">
                        <span class="hud-card-title">&#128663; Traffic &amp; Road Furniture</span>
                    </div>
                    <div class="hud-card-body">
                        <div class="tag-lattice">
                            <span class="lattice-tag">Driving Side: <strong>${ds.driving_side || 'Unknown'}</strong> (${Math.round((ds.confidence || 0) * 100)}%)</span>
                            <span class="lattice-tag">Road Lines: <strong>${rm.line_color || 'Unknown'}</strong></span>
                            <span class="lattice-tag">Center Mark: <strong>${rm.center_mark_type || 'Unknown'}</strong></span>
                            <span class="lattice-tag">Utility Pole: <strong>${(up.dominant_type || 'Unknown').replace(/_/g, ' ')}</strong></span>
                        </div>
                    </div>
                </div>

                <div class="hud-card">
                    <div class="hud-card-header">
                        <span class="hud-card-title">&#127760; Soil, Biome &amp; Solar</span>
                    </div>
                    <div class="hud-card-body">
                        <div class="tag-lattice">
                            <span class="lattice-tag">Soil Type: <strong>${sb.soil_type || 'Unknown'}</strong></span>
                            <span class="lattice-tag">Biome: <strong>${sb.vegetation_biome || 'Unknown'}</strong></span>
                            <span class="lattice-tag">Hemisphere: <strong>${ss.inferred_hemisphere || 'Unknown'}</strong></span>
                            <span class="lattice-tag">Plate Format: <strong>${lp.format || 'Unknown'}</strong></span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // -----------------------------------------------------------------------
    // Interactive Leaflet Map with Basemap Switcher
    // -----------------------------------------------------------------------
    function renderInteractiveMap(lat, lon, data) {
        const mapContainer = document.getElementById('mapContainer');
        if (!mapContainer) return;
        mapContainer.style.display = '';

        if (leafletMap) {
            leafletMap.remove();
            leafletMap = null;
        }

        const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 19,
        });

        const esriLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
            maxZoom: 19,
        });

        leafletMap = L.map('map', {
            center: [lat, lon],
            zoom: 16,
            layers: [osmLayer],
        });

        const baseLayers = {
            "OpenStreetMap (Vectors)": osmLayer,
            "ESRI World Imagery (Satellite)": esriLayer,
        };

        L.control.layers(baseLayers).addTo(leafletMap);

        // High-contrast primary target pin
        const targetIcon = L.divIcon({
            className: 'custom-map-pin',
            html: '<div class="pin-inner">&#127919;</div>',
            iconSize: [36, 36],
            iconAnchor: [18, 18],
        });

        const marker = L.marker([lat, lon], { icon: targetIcon }).addTo(leafletMap);
        marker.bindPopup(`
            <div style="font-family: sans-serif; font-size: 13px;">
                <b style="color:#d29922;">Target Ground Pin</b><br>
                <b>Coords:</b> ${lat.toFixed(6)}, ${lon.toFixed(6)}<br>
                <b>Confidence:</b> ${((data.best_estimate?.confidence || 0.85) * 100).toFixed(1)}%<br>
                <a href="https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${lat},${lon}" target="_blank" style="color:#58a6ff;">Open Street View &rarr;</a>
            </div>
        `).openPopup();

        // If candidate intersections exist, plot them
        const intersections = data.street_targeting?.candidates || [];
        intersections.slice(1, 5).forEach((cand, idx) => {
            if (cand.latitude && cand.longitude) {
                L.circleMarker([cand.latitude, cand.longitude], {
                    radius: 7,
                    color: '#58a6ff',
                    fillColor: '#58a6ff',
                    fillOpacity: 0.7,
                }).addTo(leafletMap).bindPopup(`
                    <b>Candidate Intersection #${idx + 2}</b><br>
                    ${cand.intersection || cand.name || 'Road Crossing'}<br>
                    Score: ${(cand.score || 0).toFixed(2)}
                `);
            }
        });
    }

    // -----------------------------------------------------------------------
    // Dossier Exporter
    // -----------------------------------------------------------------------
    async function exportDossier(format) {
        if (!selectedImageFile) return;
        try {
            const fd = new FormData();
            fd.append('image', selectedImageFile);
            fd.append('format', format);

            const resp = await fetch('/api/dossier', { method: 'POST', body: fd });
            const data = await resp.json();

            if (data.status === 'success') {
                const content = data.report_content;
                const ext = format === 'markdown' ? 'md' : 'html';
                const mime = format === 'markdown' ? 'text/markdown' : 'text/html';
                const blob = new Blob([content], { type: mime });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `geovision_dossier_${Date.now()}.${ext}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            } else {
                alert(`Export failed: ${data.error || 'Unknown error'}`);
            }
        } catch (e) {
            alert(`Dossier export failed: ${e.message}`);
        }
    }

})();
