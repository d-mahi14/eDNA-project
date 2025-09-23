document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const dnaInput = document.getElementById('dna-input');
    const statusEl = document.getElementById('status');
    const resultsSection = document.getElementById('results-section');

    // KPI Elements
    const richnessEl = document.getElementById('kpi-richness');
    const shannonEl = document.getElementById('kpi-shannon');
    const simpsonEl = document.getElementById('kpi-simpson');
    const pielouEl = document.getElementById('kpi-pielou');
    const novelEl = document.getElementById('kpi-novel');

    // Summary
    const taxaRowsEl = document.getElementById('taxa-rows');
    const summaryIdentifiedEl = document.getElementById('summary-identified');
    const summaryNoiseEl = document.getElementById('summary-noise');

    // Plots
    const pcaPlotImg = document.getElementById('pca-plot-img');
    const kDistancePlotImg = document.getElementById('k-distance-plot-img');

    analyzeBtn.addEventListener('click', handleAnalysis);

    async function handleAnalysis() {
        const sequences = dnaInput.value;
        if (!sequences.trim()) {
            showStatus('Please input DNA sequences.', 'error');
            return;
        }

        setLoadingState(true);
        showStatus('Analyzing... this may take a moment.', 'loading');
        
        try {
            // API Call to Backend
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ sequence: sequences })
            });

            const result = await response.json();

            if (!response.ok) {
                // Handle server-side errors
                throw new Error(result.error || `HTTP error! status: ${response.status}`);
            }
            
            // Update UI with Results
            updateUI(result);
            showStatus('Analysis complete.', 'success');
            // Scroll to results after a short delay
            setTimeout(() => {
                 resultsSection.scrollIntoView({ behavior: 'smooth' });
            }, 100);


        } catch (error) {
            console.error('Analysis failed:', error);
            showStatus(`Error: ${error.message}`, 'error');
        } finally {
            setLoadingState(false);
        }
    }

    //UI Update Function
    function updateUI(data) {
        if (!data || !data.summary) {
            console.error('Invalid data structure received from server');
            showStatus('Failed to parse results.', 'error');
            return;
        }

        const summary = data.summary;
        const metrics = summary.biodiversity_metrics;
        const taxaSummary = summary.taxonomic_summary;
        const taxaDetails = summary.taxa_details;
        const plots = data.plots;

        // Update KPIs
        richnessEl.textContent = metrics.species_richness || '0';
        shannonEl.textContent = metrics.shannon_diversity_index || '0';
        simpsonEl.textContent = metrics.simpson_diversity_index || '0';
        pielouEl.textContent = metrics.pielou_evenness_index || '0';
        novelEl.textContent = taxaSummary.novel_taxa || '0';
        
        // Update summary
        summaryIdentifiedEl.textContent = taxaSummary.identified_taxa || '0';
        summaryNoiseEl.textContent = taxaSummary.noise_points || '0';

        taxaRowsEl.innerHTML = ''; // Clear previous results
        if(taxaDetails && taxaDetails.length > 0) {
            taxaDetails.forEach(item => {
                const row = document.createElement('tr');
                // Format confidence as percentage
                const confidence = item.status === 'Novel' ? 'N/A' : `${(item.confidence * 100).toFixed(1)}%`;
                
                row.innerHTML = `
                    <td>${item.cluster_id}</td>
                    <td><span class="status-pill ${item.status.toLowerCase()}">${item.status}</span></td>
                    <td>${item.abundance}</td>
                    <td>${confidence}</td>
                    <td>${item.taxon_name}</td>
                    <td>${item.taxonomic_level}</td> 
                `; // <-- FIXED: Changed item.lineage to item.taxonomic_level
                taxaRowsEl.appendChild(row);
            });
        } else {
             taxaRowsEl.innerHTML = '<tr><td colspan="6">No taxa details found.</td></tr>';
        }


        // Update plot images with base64 data
        if(plots.pca_scatter) {
            pcaPlotImg.src = `data:image/png;base64,${plots.pca_scatter}`;
        }
        if(plots.k_distance) {
            kDistancePlotImg.src = `data:image/png;base64,${plots.k_distance}`;
        }
    }

    function showStatus(message, type = 'loading') {
        statusEl.textContent = message;
        statusEl.className = 'status'; // Reset classes
        if (type === 'error') {
            statusEl.classList.add('error');
        } else if (type === 'success') {
            statusEl.classList.add('success');
        }
    }

    function setLoadingState(isLoading) {
        analyzeBtn.disabled = isLoading;
        dnaInput.disabled = isLoading;
    }

    //Logic for Collapsible Section
    const collapseHeaders = document.querySelectorAll('.collapse-header');
    collapseHeaders.forEach(header => {
        header.addEventListener('click', () => {
            const contentId = header.getAttribute('data-collapse');
            const content = document.getElementById(contentId);
            
            if (content) {
                header.classList.toggle('open');
                content.classList.toggle('hidden');
            }
        });
    });
});





