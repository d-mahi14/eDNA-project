document.addEventListener('DOMContentLoaded', () => {
    // --- Global State ---
    let noiseDetailsData = [];

    // --- Element Selectors ---
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadSampleBtn = document.getElementById('load-sample-btn');
    const dnaInput = document.getElementById('dna-input');
    const statusEl = document.getElementById('status');
    const resultsSection = document.getElementById('results-section');
    const analyzeBtnText = document.getElementById('analyze-btn-text');
    const analyzeIcon = document.getElementById('analyze-icon');
    const analyzeLoadingSpinner = analyzeBtn.querySelector('#loading-spinner');
    const loadBtnText = document.getElementById('load-btn-text');
    const loadIcon = document.getElementById('load-icon');
    const loadLoadingSpinner = loadSampleBtn.querySelector('#load-spinner');
    const richnessEl = document.getElementById('kpi-richness');
    const shannonEl = document.getElementById('kpi-shannon');
    const simpsonEl = document.getElementById('kpi-simpson');
    const pielouEl = document.getElementById('kpi-pielou');
    const novelEl = document.getElementById('kpi-novel');
    const taxaRowsEl = document.getElementById('taxa-rows');
    const summaryIdentifiedEl = document.getElementById('summary-identified');
    const summaryNovelEl = document.getElementById('summary-novel');
    const summaryNoiseEl = document.getElementById('summary-noise');
    const tablePlaceholder = document.getElementById('table-placeholder');
    const clusterPlotImg = document.getElementById('cluster-plot-img');
    const plotModal = document.getElementById('plot-modal');
    const modalImg = document.getElementById('modal-plot-img');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const expandPlotBtns = document.querySelectorAll('.expand-btn');
    
    // --- Noise Modal Elements ---
    const noisePill = document.getElementById('noise-pill');
    const noiseModal = document.getElementById('noise-modal');
    const noiseModalCloseBtn = document.getElementById('noise-modal-close-btn');
    const noiseModalRowsEl = document.getElementById('noise-modal-rows');

    // --- Event Listeners ---
    analyzeBtn.addEventListener('click', handleAnalysis);
    loadSampleBtn.addEventListener('click', handleLoadSample);

    expandPlotBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const sourceImg = document.getElementById(e.currentTarget.dataset.plotTarget);
            if (sourceImg && sourceImg.src && !sourceImg.classList.contains('hidden')) {
                modalImg.src = sourceImg.src;
                plotModal.classList.remove('hidden');
            }
        });
    });

    const closeModal = (modal) => modal.classList.add('hidden');
    modalCloseBtn.addEventListener('click', () => closeModal(plotModal));
    plotModal.addEventListener('click', (e) => e.target === plotModal && closeModal(plotModal));
    
    noisePill.addEventListener('click', () => {
        if (noiseDetailsData.length > 0) {
            populateNoiseModal(noiseDetailsData);
            noiseModal.classList.remove('hidden');
        }
    });
    noiseModalCloseBtn.addEventListener('click', () => closeModal(noiseModal));
    noiseModal.addEventListener('click', (e) => e.target === noiseModal && closeModal(noiseModal));


    // --- Core Functions ---
    async function handleLoadSample() {
        setLoadingState(true, 'load');
        showStatus('Loading sample data...', 'loading');
        try {
            const response = await fetch('/generate_mock_data');
            if (!response.ok) throw new Error((await response.json()).error || 'Failed to fetch sample data.');
            const result = await response.json();
            dnaInput.value = result.sequences;
            showStatus('Sample data loaded.', 'success');
        } catch (error) {
            showStatus(`Error: ${error.message}`, 'error');
        } finally {
            setLoadingState(false, 'load');
        }
    }
    
    async function handleAnalysis() {
        const sequences = dnaInput.value.trim();
        if (!sequences) {
            showStatus('Please input DNA sequences.', 'error');
            return;
        }

        setLoadingState(true, 'analyze');
        showStatus('Analyzing... this may take a moment.', 'loading');
        resultsSection.classList.add('hidden');

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sequence: sequences })
            });

            if (!response.ok) throw new Error((await response.json()).error || `HTTP error! Status: ${response.status}`);
            
            const result = await response.json();
            updateUI(result);
            showStatus('Analysis complete.', 'success');
            resultsSection.classList.remove('hidden');
            setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);

        } catch (error) {
            showStatus(`Error: ${error.message}`, 'error');
        } finally {
            setLoadingState(false, 'analyze');
        }
    }

    function updateUI(data) {
        if (!data || !data.summary) {
            showStatus('Failed to parse results.', 'error');
            return;
        }

        const { summary, plots } = data;
        const { biodiversity_metrics, taxonomic_summary, taxa_details, noise_point_details } = summary;

        noiseDetailsData = noise_point_details || [];

        // Update KPIs
        richnessEl.textContent = biodiversity_metrics.species_richness || '0';
        shannonEl.textContent = (biodiversity_metrics.shannon_diversity_index || 0).toFixed(2);
        simpsonEl.textContent = (biodiversity_metrics.simpson_diversity_index || 0).toFixed(2);
        pielouEl.textContent = (biodiversity_metrics.pielou_evenness_index || 0).toFixed(2);
        novelEl.textContent = taxonomic_summary.novel_taxa || '0';

        // Update summary pills
        summaryIdentifiedEl.textContent = taxonomic_summary.identified_taxa || '0';
        summaryNovelEl.textContent = taxonomic_summary.novel_taxa || '0';
        summaryNoiseEl.textContent = taxonomic_summary.noise_points || '0';
        noisePill.style.cursor = (taxonomic_summary.noise_points > 0) ? 'pointer' : 'default';

        // Update results table
        taxaRowsEl.innerHTML = '';
        if (taxa_details && taxa_details.length > 0) {
            tablePlaceholder.classList.add('hidden');
            taxa_details.forEach(item => {
                const row = document.createElement('tr');
                const confidence = item.status === 'Novel' ? 'N/A' : `${(item.confidence * 100).toFixed(1)}%`;
                const statusClass = item.status.toLowerCase();

                row.innerHTML = `
                    <td>${item.cluster_id}</td>
                    <td><span class="status-pill ${statusClass}">${item.status}</span></td>
                    <td>${item.abundance}</td>
                    <td>${confidence}</td>
                    <td>${item.taxon_name}</td>
                    <td>${item.taxonomic_level}</td> 
                `;
                taxaRowsEl.appendChild(row);
            });
        } else {
            tablePlaceholder.classList.remove('hidden');
            tablePlaceholder.textContent = 'No distinct clusters were identified.';
        }

        if (plots && plots.cluster_plot) {
            clusterPlotImg.src = `data:image/png;base64,${plots.cluster_plot}`;
            clusterPlotImg.classList.remove('hidden');
        } else {
            clusterPlotImg.classList.add('hidden');
        }
    }

    function populateNoiseModal(noiseData) {
        noiseModalRowsEl.innerHTML = '';
        if (noiseData.length === 0) {
            noiseModalRowsEl.innerHTML = '<tr><td colspan="4" class="text-center">No noise points found.</td></tr>';
            return;
        }
        noiseData.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${item.noise_point_id}</td>
                <td>${item.closest_known_cluster_id}</td>
                <td>${item.closest_taxon_name}</td>
                <td>${item.distance}</td>
            `;
            noiseModalRowsEl.appendChild(row);
        });
    }

    function showStatus(message, type = 'loading') {
        statusEl.textContent = message;
        statusEl.className = 'status';
        statusEl.classList.add(type);
    }

    function setLoadingState(isLoading, buttonType) {
        const btn = (buttonType === 'analyze') ? analyzeBtn : loadSampleBtn;
        const otherBtn = (buttonType === 'analyze') ? loadSampleBtn : analyzeBtn;
        const icon = (buttonType === 'analyze') ? analyzeIcon : loadIcon;
        const spinner = (buttonType === 'analyze') ? analyzeLoadingSpinner : loadLoadingSpinner;
        const textEl = (buttonType === 'analyze') ? analyzeBtnText : loadBtnText;
        const text = (buttonType === 'analyze') ? 'Analyze' : 'Load Sample Data';
        
        btn.disabled = isLoading;
        otherBtn.disabled = isLoading;
        dnaInput.disabled = isLoading;

        if (isLoading) {
            icon.classList.add('hidden');
            spinner.classList.remove('hidden');
            textEl.textContent = (buttonType === 'analyze') ? 'Analyzing...' : 'Loading...';
        } else {
            icon.classList.remove('hidden');
            spinner.classList.add('hidden');
            textEl.textContent = text;
        }
    }
});

