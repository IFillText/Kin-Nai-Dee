// ========================================================
// KIN NAI DEE - RATING VERIFICATION & LATENCY INSPECTOR
// ========================================================

document.addEventListener('DOMContentLoaded', () => {
  // Input elements
  const reviewInput = document.getElementById('review-input');
  const inputFeedback = document.getElementById('input-feedback');
  const charCount = document.getElementById('char-count');
  const starButtons = document.querySelectorAll('.star-btn');
  const btnAnalyze = document.getElementById('btn-analyze');
  const btnClear = document.getElementById('btn-clear');
  const presetChips = document.querySelectorAll('.preset-chip');

  // Output container elements
  const emptyState = document.getElementById('empty-state');
  const resultBody = document.getElementById('result-body');
  const statusTag = document.getElementById('status-tag');

  // Verdict elements
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictDesc = document.getElementById('verdict-desc');

  // Decision & Taste Matrix elements
  const decisionVerdictText = document.getElementById('decision-verdict-text');
  const tasteSpicyVal = document.getElementById('taste-spicy-val');
  const tasteSpicyBar = document.getElementById('taste-spicy-bar');
  const tasteSweetVal = document.getElementById('taste-sweet-val');
  const tasteSweetBar = document.getElementById('taste-sweet-bar');
  const tasteRichVal = document.getElementById('taste-rich-val');
  const tasteRichBar = document.getElementById('taste-rich-bar');
  const tasteSourVal = document.getElementById('taste-sour-val');
  const tasteSourBar = document.getElementById('taste-sour-bar');

  const aspectValuePill = document.getElementById('aspect-value-pill');
  const aspectValueStatus = document.getElementById('aspect-value-status');
  const aspectTimePill = document.getElementById('aspect-time-pill');
  const aspectTimeStatus = document.getElementById('aspect-time-status');
  const aspectServicePill = document.getElementById('aspect-service-pill');
  const aspectServiceStatus = document.getElementById('aspect-service-status');
  const aspectPlacePill = document.getElementById('aspect-place-pill');
  const aspectPlaceStatus = document.getElementById('aspect-place-status');

  // BiLSTM elements
  const bilstmLatency = document.getElementById('bilstm-latency');
  const actualStarDisplay = document.getElementById('actual-star-display');
  const bilstmStarDisplay = document.getElementById('bilstm-star-display');
  const bilstmMatchBadge = document.getElementById('bilstm-match-badge');
  const bilstmConf = document.getElementById('bilstm-conf');
  const bilstmProbBars = document.getElementById('bilstm-prob-bars');

  // LinearSVC elements
  const baseLatency = document.getElementById('base-latency');
  const baseActualStarDisplay = document.getElementById('base-actual-star-display');
  const baseStarDisplay = document.getElementById('base-star-display');
  const baseMatchBadge = document.getElementById('base-match-badge');
  const baseConf = document.getElementById('base-conf');
  const baseProbBars = document.getElementById('base-prob-bars');

  // Tokens
  const tokenCountBadge = document.getElementById('token-count-badge');
  const tokensList = document.getElementById('tokens-list');

  // History & Metrics elements
  const metricTotalRuns = document.getElementById('metric-total-runs');
  const metricBilstmAcc = document.getElementById('metric-bilstm-acc');
  const metricBilstmMeanLat = document.getElementById('metric-bilstm-mean-lat');
  const metricBaseMeanLat = document.getElementById('metric-base-mean-lat');
  const historyTbody = document.getElementById('history-tbody');
  const emptyHistoryRow = document.getElementById('empty-history-row');
  const btnClearHistory = document.getElementById('btn-clear-history');

  let selectedStar = 1;
  let presetsData = {};
  const historyRecords = [];

  // ----------------------------------------------------
  // 1. Star Rating Selection
  // ----------------------------------------------------
  starButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      starButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedStar = parseInt(btn.getAttribute('data-star'), 10);
    });
  });

  function setStarRating(starNum) {
    selectedStar = starNum;
    starButtons.forEach(b => {
      if (parseInt(b.getAttribute('data-star'), 10) === starNum) {
        b.classList.add('active');
      } else {
        b.classList.remove('active');
      }
    });
  }

  // ----------------------------------------------------
  // 2. Textarea Character Count & Input Reset
  // ----------------------------------------------------
  reviewInput.addEventListener('input', () => {
    const len = reviewInput.value.length;
    charCount.textContent = `${len.toLocaleString()} ตัวอักษร`;
    if (inputFeedback) {
      inputFeedback.style.display = 'none';
      reviewInput.classList.remove('input-invalid');
    }
  });

  // ----------------------------------------------------
  // 3. Load Presets
  // ----------------------------------------------------
  fetch('/api/presets')
    .then(res => res.json())
    .then(data => {
      data.forEach(item => {
        presetsData[item.id] = item;
      });
    })
    .catch(err => console.error('Error fetching presets:', err));

  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const presetId = chip.getAttribute('data-preset');
      const item = presetsData[presetId];
      if (item) {
        reviewInput.value = item.text;
        charCount.textContent = `${item.text.length.toLocaleString()} ตัวอักษร`;
        if (inputFeedback) {
          inputFeedback.style.display = 'none';
          reviewInput.classList.remove('input-invalid');
        }
        setStarRating(item.star);
        runAnalysis();
      }
    });
  });

  // ----------------------------------------------------
  // 4. Taste & Aspect Helpers
  // ----------------------------------------------------
  function updateAspectPill(pillEl, statusEl, aspectData, defaultLabel) {
    if (!pillEl || !statusEl) return;
    pillEl.classList.remove('positive', 'negative', 'neutral');
    if (aspectData && aspectData.status) {
      pillEl.classList.add(aspectData.status);
      statusEl.textContent = aspectData.summary || aspectData.label || defaultLabel;
    } else {
      pillEl.classList.add('neutral');
      statusEl.textContent = defaultLabel;
    }
  }

  function renderDecisionInsights(insights) {
    if (!insights) return;

    // 1. Verdict Summary Box
    const verdict = insights.decision_verdict || insights.verdict;
    if (decisionVerdictText && verdict) {
      decisionVerdictText.textContent = verdict;
    }

    // 2. Taste Meters
    if (insights.taste) {
      const spicy = insights.taste.spiciness || insights.taste.spicy;
      const sweet = insights.taste.sweetness || insights.taste.sweet;
      const rich = insights.taste.richness || insights.taste.rich;
      const sour = insights.taste.sourness || insights.taste.sour;

      if (spicy) {
        const pct = spicy.score_pct !== undefined ? spicy.score_pct : (spicy.percent || 0);
        if (tasteSpicyVal) tasteSpicyVal.textContent = spicy.label || 'ไม่ระบุ';
        if (tasteSpicyBar) tasteSpicyBar.style.width = `${pct}%`;
      }
      if (sweet) {
        const pct = sweet.score_pct !== undefined ? sweet.score_pct : (sweet.percent || 0);
        if (tasteSweetVal) tasteSweetVal.textContent = sweet.label || 'ไม่ระบุ';
        if (tasteSweetBar) tasteSweetBar.style.width = `${pct}%`;
      }
      if (rich) {
        const pct = rich.score_pct !== undefined ? rich.score_pct : (rich.percent || 0);
        if (tasteRichVal) tasteRichVal.textContent = rich.label || 'ไม่ระบุ';
        if (tasteRichBar) tasteRichBar.style.width = `${pct}%`;
      }
      if (sour) {
        const pct = sour.score_pct !== undefined ? sour.score_pct : (sour.detected ? 75 : 0);
        if (tasteSourVal) tasteSourVal.textContent = sour.label || 'ไม่ระบุ';
        if (tasteSourBar) tasteSourBar.style.width = `${pct}%`;
      }
    }

    // 3. Decision Factors (Aspects)
    if (insights.aspects) {
      updateAspectPill(aspectValuePill, aspectValueStatus, insights.aspects.value, 'ราคาและปริมาณตามมาตรฐาน');
      updateAspectPill(aspectTimePill, aspectTimeStatus, insights.aspects.time, 'เวลารออาหารปกติ');
      updateAspectPill(aspectServicePill, aspectServiceStatus, insights.aspects.service, 'การบริการตามมาตรฐาน');
      updateAspectPill(aspectPlacePill, aspectPlaceStatus, insights.aspects.place, 'บรรยากาศทั่วไป');
    }
  }

  function resetDecisionInsights() {
    if (decisionVerdictText) {
      decisionVerdictText.textContent = 'รอการวิเคราะห์รีวิวเพื่อประเมินรสชาติและปัจจัยการตัดสินใจ...';
    }
    if (tasteSpicyVal) tasteSpicyVal.textContent = 'ไม่ระบุ';
    if (tasteSpicyBar) tasteSpicyBar.style.width = '0%';
    if (tasteSweetVal) tasteSweetVal.textContent = 'ไม่ระบุ';
    if (tasteSweetBar) tasteSweetBar.style.width = '0%';
    if (tasteRichVal) tasteRichVal.textContent = 'ไม่ระบุ';
    if (tasteRichBar) tasteRichBar.style.width = '0%';
    if (tasteSourVal) tasteSourVal.textContent = 'ไม่ระบุ';
    if (tasteSourBar) tasteSourBar.style.width = '0%';

    updateAspectPill(aspectValuePill, aspectValueStatus, null, 'ราคาและปริมาณตามมาตรฐาน');
    updateAspectPill(aspectTimePill, aspectTimeStatus, null, 'เวลารออาหารปกติ');
    updateAspectPill(aspectServicePill, aspectServiceStatus, null, 'การบริการตามมาตรฐาน');
    updateAspectPill(aspectPlacePill, aspectPlaceStatus, null, 'บรรยากาศทั่วไป');
  }

  // ----------------------------------------------------
  // 5. Clear Input & State
  // ----------------------------------------------------
  btnClear.addEventListener('click', () => {
    reviewInput.value = '';
    charCount.textContent = '0 ตัวอักษร';
    if (inputFeedback) {
      inputFeedback.style.display = 'none';
      reviewInput.classList.remove('input-invalid');
    }
    setStarRating(1);
    emptyState.style.display = 'block';
    resultBody.style.display = 'none';
    statusTag.textContent = 'รอการตรวจสอบ';
    statusTag.className = 'eval-tag';
    resetDecisionInsights();
  });

  // ----------------------------------------------------
  // 6. Probability Bar Renderer
  // ----------------------------------------------------
  function renderProbBars(container, probs) {
    container.innerHTML = '';
    probs.forEach((p, idx) => {
      const starLevel = idx + 1;
      const row = document.createElement('div');
      row.className = 'prob-bar-row';
      row.innerHTML = `
        <span class="prob-bar-label">${starLevel}★</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${p}%"></div>
        </div>
        <span class="prob-bar-value">${p.toFixed(1)}%</span>
      `;
      container.appendChild(row);
    });
  }

  // ----------------------------------------------------
  // 7. Analysis Execution
  // ----------------------------------------------------
  btnAnalyze.addEventListener('click', runAnalysis);

  function runAnalysis() {
    const text = reviewInput.value.trim();
    if (!text) {
      if (inputFeedback) {
        inputFeedback.textContent = 'กรุณาระบุข้อความรีวิวก่อนเริ่มการตรวจสอบ';
        inputFeedback.style.display = 'block';
      }
      reviewInput.classList.add('input-invalid');
      reviewInput.focus();
      return;
    }

    if (inputFeedback) {
      inputFeedback.style.display = 'none';
    }
    reviewInput.classList.remove('input-invalid');

    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = `
      <span class="btn-text">กำลังประมวลผล & วัด Latency...</span>
      <span class="btn-icon-bubble" aria-hidden="true">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="9" stroke-dasharray="28" stroke-dashoffset="10"></circle>
        </svg>
      </span>
    `;
    statusTag.textContent = 'กำลังประมวลผล...';

    fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: text,
        given_star: selectedStar
      })
    })
      .then(res => {
        if (!res.ok) throw new Error('การประมวลผลล้มเหลว');
        return res.json();
      })
      .then(data => {
        // Show result, hide empty state
        emptyState.style.display = 'none';
        resultBody.style.display = 'block';

        const actualStar = data.given_star;
        const biPred = data.proposed_bilstm.predicted_star;
        const basePred = data.baseline_linearsvc.predicted_star;

        const isBiCorrect = biPred === actualStar;
        const isBaseCorrect = basePred === actualStar;

        // 1. Verdict Banner
        verdictBanner.className = 'verdict-banner';
        if (isBiCorrect) {
          verdictBanner.classList.add('correct');
          verdictTitle.textContent = `สอดคล้องสมบูรณ์: โมเดลทำนาย ${biPred}★ ตรงกับดาวจริง ${actualStar}★`;
          statusTag.textContent = 'สอดคล้องปกติ';
          statusTag.className = 'eval-tag text-success';
        } else {
          const diff = Math.abs(biPred - actualStar);
          if (diff === 1) {
            verdictBanner.classList.add('slight');
            verdictTitle.textContent = `คลาดเคลื่อนเล็กน้อย ±1 ระดับ (ดาวจริง: ${actualStar}★ → ทำนาย: ${biPred}★)`;
            statusTag.textContent = 'คลาดเคลื่อน ±1★';
            statusTag.className = 'eval-tag text-warning';
          } else {
            verdictBanner.classList.add('wrong');
            verdictTitle.textContent = `ตรวจพบความขัดแย้งรุนแรง (ดาวจริง: ${actualStar}★ → ทำนาย: ${biPred}★ ต่างกัน ${diff} ดาว)`;
            statusTag.textContent = 'ขัดแย้งรุนแรง';
            statusTag.className = 'eval-tag text-danger';
          }
        }
        verdictDesc.textContent = data.consistency.detail || `คะแนนจริง: ${actualStar} ดาว | BiLSTM ทายได้: ${biPred} ดาว | LinearSVC ทายได้: ${basePred} ดาว`;

        // 2. Decision Insights & Taste Profile
        if (data.decision_insights) {
          renderDecisionInsights(data.decision_insights);
        }

        // 3. Proposed BiLSTM Card
        const bi = data.proposed_bilstm;
        bilstmLatency.textContent = `${bi.latency_ms.toFixed(2)} ms`;
        actualStarDisplay.textContent = `${actualStar}★`;
        bilstmStarDisplay.textContent = `${biPred}★`;
        
        if (isBiCorrect) {
          bilstmMatchBadge.className = 'match-badge correct';
          bilstmMatchBadge.textContent = 'สอดคล้อง (Match)';
        } else {
          bilstmMatchBadge.className = 'match-badge wrong';
          bilstmMatchBadge.textContent = `ต่างกัน ${Math.abs(biPred - actualStar)}★`;
        }
        bilstmConf.textContent = `ความมั่นใจ: ${bi.confidence.toFixed(1)}% | Latency: ${bi.latency_ms.toFixed(2)} ms`;
        renderProbBars(bilstmProbBars, bi.probabilities);

        // 4. Baseline LinearSVC Card
        const base = data.baseline_linearsvc;
        baseLatency.textContent = `${base.latency_ms.toFixed(2)} ms`;
        baseActualStarDisplay.textContent = `${actualStar}★`;
        baseStarDisplay.textContent = `${basePred}★`;

        if (isBaseCorrect) {
          baseMatchBadge.className = 'match-badge correct';
          baseMatchBadge.textContent = 'สอดคล้อง (Match)';
        } else {
          baseMatchBadge.className = 'match-badge wrong';
          baseMatchBadge.textContent = `ต่างกัน ${Math.abs(basePred - actualStar)}★`;
        }
        baseConf.textContent = `คะแนนสัมพัทธ์: ${base.confidence.toFixed(1)}% | Latency: ${base.latency_ms.toFixed(2)} ms`;
        renderProbBars(baseProbBars, base.probabilities);

        // 5. Token Breakdown
        tokenCountBadge.textContent = `${data.token_count} คำ`;
        tokensList.innerHTML = '';
        data.tokens.forEach(item => {
          const chip = document.createElement('span');
          chip.className = `token-chip ${item.type}`;
          chip.textContent = item.token;
          tokensList.appendChild(chip);
        });

        // 6. Append Record to Latency & History Table
        addHistoryRecord({
          text: text,
          actualStar: actualStar,
          biPred: biPred,
          isBiCorrect: isBiCorrect,
          biLat: bi.latency_ms,
          basePred: basePred,
          isBaseCorrect: isBaseCorrect,
          baseLat: base.latency_ms
        });
      })
      .catch(err => {
        console.error(err);
        if (inputFeedback) {
          inputFeedback.textContent = 'เกิดข้อผิดพลาดในการประมวลผลโมเดล กรุณาลองใหม่อีกครั้ง';
          inputFeedback.style.display = 'block';
        }
      })
      .finally(() => {
        btnAnalyze.disabled = false;
        btnAnalyze.innerHTML = `
          <span class="btn-text">ตรวจสอบการทำนาย & วัด Latency</span>
          <span class="btn-icon-bubble" aria-hidden="true">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
            </svg>
          </span>
        `;
      });
  }

  // ----------------------------------------------------
  // 8. History & Latency Log Manager
  // ----------------------------------------------------
  function addHistoryRecord(record) {
    historyRecords.unshift(record); // Add to beginning

    if (emptyHistoryRow) {
      emptyHistoryRow.style.display = 'none';
    }

    // Re-render table rows
    historyTbody.innerHTML = '';
    historyRecords.forEach((rec, idx) => {
      const rowNum = historyRecords.length - idx;
      const tr = document.createElement('tr');
      const truncatedText = rec.text.length > 55 ? rec.text.substring(0, 52) + '...' : rec.text;

      tr.innerHTML = `
        <td><strong>#${rowNum}</strong></td>
        <td title="${rec.text.replace(/"/g, '&quot;')}">${truncatedText}</td>
        <td><strong>${rec.actualStar}★</strong></td>
        <td><strong>${rec.biPred}★</strong></td>
        <td>
          <span class="table-badge ${rec.isBiCorrect ? 'correct' : 'wrong'}">
            ${rec.isBiCorrect ? 'สอดคล้อง' : 'ขัดแย้ง'}
          </span>
        </td>
        <td class="table-lat text-accent">${rec.biLat.toFixed(2)} ms</td>
        <td><strong>${rec.basePred}★</strong></td>
        <td>
          <span class="table-badge ${rec.isBaseCorrect ? 'correct' : 'wrong'}">
            ${rec.isBaseCorrect ? 'สอดคล้อง' : 'ขัดแย้ง'}
          </span>
        </td>
        <td class="table-lat text-info">${rec.baseLat.toFixed(2)} ms</td>
      `;
      historyTbody.appendChild(tr);
    });

    // Update Summary Metrics
    const total = historyRecords.length;
    const biCorrectCount = historyRecords.filter(r => r.isBiCorrect).length;
    const biAcc = (biCorrectCount / total) * 100;
    const biMeanLat = historyRecords.reduce((acc, r) => acc + r.biLat, 0) / total;
    const baseMeanLat = historyRecords.reduce((acc, r) => acc + r.baseLat, 0) / total;

    metricTotalRuns.textContent = total.toLocaleString();
    metricBilstmAcc.textContent = `${biAcc.toFixed(1)}% (${biCorrectCount}/${total})`;
    metricBilstmMeanLat.textContent = `${biMeanLat.toFixed(2)} ms`;
    metricBaseMeanLat.textContent = `${baseMeanLat.toFixed(2)} ms`;
  }

  btnClearHistory.addEventListener('click', () => {
    historyRecords.length = 0;
    historyTbody.innerHTML = `
      <tr id="empty-history-row">
        <td colspan="9" class="text-center text-muted py-4" style="text-align: center; padding: 24px; color: var(--text-muted);">
          ยังไม่มีประวัติการทดสอบ — เริ่มทดสอบด้านบนเพื่อบันทึกข้อมูล
        </td>
      </tr>
    `;
    metricTotalRuns.textContent = '0';
    metricBilstmAcc.textContent = '0%';
    metricBilstmMeanLat.textContent = '0.00 ms';
    metricBaseMeanLat.textContent = '0.00 ms';
  });

});
