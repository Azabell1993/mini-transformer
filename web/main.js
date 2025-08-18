// Mini-Transformer Web Interface
// 전역 변수
let isGenerating = false;
let currentTokens = [];

// DOM 로드 완료 시 초기화
document.addEventListener('DOMContentLoaded', () => {
  initializeApp();
});

// 앱 초기화
function initializeApp() {
  initBackToTop();
  checkServerStatus();
  loadConfigFromServer();
  setupEventListeners();
  console.log('🚀 Mini-Transformer 웹 인터페이스가 초기화되었습니다.');
}

// 스크롤 탑 버튼 초기화
function initBackToTop() {
  const btn = document.getElementById('backToTop');
  
  window.addEventListener('scroll', () => {
    btn.style.display = window.scrollY > 200 ? 'block' : 'none';
  });
  
  btn.addEventListener('click', () => {
    window.scrollTo({ 
      top: 0, 
      behavior: 'smooth' 
    });
  });
}

// 서버 상태 확인
async function checkServerStatus() {
  const statusDot = document.getElementById('serverStatus');
  const statusText = document.getElementById('statusText');
  
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: 'tokens=1'
    });
    
    if (response.ok) {
      statusDot.className = 'status-dot online';
      statusText.textContent = '서버 연결됨';
    } else {
      throw new Error('Server response not ok');
    }
  } catch (error) {
    statusDot.className = 'status-dot offline';
    statusText.textContent = '서버 연결 실패';
    console.warn('⚠️ 서버 연결 실패:', error.message);
  }
}

// 서버에서 설정 정보 로드 (실제로는 기본값 사용)
function loadConfigFromServer() {
  // 실제 구현에서는 서버에서 설정을 가져올 수 있음
  const config = {
    vocab_size: 3200,
    d_model: 128,
    n_layers: 1,
    n_heads: 2,
    d_ff: 512,
    max_seq_len: 64
  };
  
  updateConfigDisplay(config);
}

// 설정 정보 화면에 업데이트
function updateConfigDisplay(config) {
  document.getElementById('vocabSize').textContent = config.vocab_size;
  document.getElementById('dModel').textContent = config.d_model;
  document.getElementById('nLayers').textContent = config.n_layers;
  document.getElementById('nHeads').textContent = config.n_heads;
  document.getElementById('dFf').textContent = config.d_ff;
  document.getElementById('maxSeq').textContent = config.max_seq_len;
  document.getElementById('maxTokenId').textContent = config.vocab_size - 1;
}

// 이벤트 리스너 설정
function setupEventListeners() {
  const tokenInput = document.getElementById('tokenInput');
  
  // 엔터키로 예측 실행
  tokenInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !isGenerating) {
      submitPrediction();
    }
  });
  
  // 입력 유효성 검사
  tokenInput.addEventListener('input', validateTokenInput);
}

// 토큰 입력 유효성 검사
function validateTokenInput() {
  const input = document.getElementById('tokenInput');
  const value = input.value.trim();
  
  // 숫자와 쉼표만 허용
  const isValid = /^[0-9,\s]*$/.test(value);
  
  if (!isValid && value !== '') {
    input.style.borderColor = 'var(--danger-color)';
    showTooltip(input, '숫자와 쉼표만 입력하세요');
  } else {
    input.style.borderColor = 'var(--gray-300)';
    hideTooltip();
  }
}

// 툴팁 표시
function showTooltip(element, message) {
  hideTooltip(); // 기존 툴팁 제거
  
  const tooltip = document.createElement('div');
  tooltip.id = 'inputTooltip';
  tooltip.textContent = message;
  tooltip.style.cssText = `
    position: absolute;
    background: var(--danger-color);
    color: white;
    padding: 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    z-index: 1000;
    margin-top: 5px;
  `;
  
  element.parentNode.appendChild(tooltip);
}

// 툴팁 숨기기
function hideTooltip() {
  const tooltip = document.getElementById('inputTooltip');
  if (tooltip) {
    tooltip.remove();
  }
}

// 예시 토큰 설정
function setTokens(tokens) {
  document.getElementById('tokenInput').value = tokens;
  validateTokenInput();
}

// 로딩 표시
function showLoading() {
  document.getElementById('loadingOverlay').style.display = 'flex';
}

// 로딩 숨기기
function hideLoading() {
  document.getElementById('loadingOverlay').style.display = 'none';
}

// 토큰 입력 파싱
function parseTokenInput(input) {
  if (!input.trim()) return null;
  
  return input.split(',')
    .map(token => token.trim())
    .filter(token => token !== '')
    .map(token => parseInt(token, 10))
    .filter(token => !isNaN(token));
}

// 버튼 상태 업데이트
function updateButtonStates(disabled) {
  document.getElementById('predictBtn').disabled = disabled;
  
  if (disabled) {
    document.getElementById('predictBtn').textContent = '🔄 예측 중...';
  } else {
    document.getElementById('predictBtn').textContent = '🎯 예측하기';
  }
}

// 예측 요청 전송
async function submitPrediction() {
  if (isGenerating) return;
  
  const input = document.getElementById('tokenInput');
  const tokens = parseTokenInput(input.value);
  
  if (!tokens || tokens.length === 0) {
    showError('유효한 토큰을 입력하세요.');
    return;
  }
  
  if (tokens.some(token => token < 0 || token >= 3200)) {
    showError('토큰은 0부터 3199 사이의 값이어야 합니다.');
    return;
  }
  
  try {
    isGenerating = true;
    currentTokens = tokens;
    updateButtonStates(true);
    showLoading();
    
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `tokens=${tokens.join(',')}`
    });
    
    if (!response.ok) {
      throw new Error(`서버 오류: ${response.status}`);
    }
    
    const result = await response.text();
    displayPredictionResult(tokens, result);
    
  } catch (error) {
    console.error('예측 오류:', error);
    showError(`예측 중 오류가 발생했습니다: ${error.message}`);
  } finally {
    isGenerating = false;
    updateButtonStates(false);
    hideLoading();
  }
}

async function generateSequence() {
  if (isGenerating) return;
  
  const input = document.getElementById('tokenInput');
  let tokens = parseTokenInput(input.value);
  
  if (!tokens || tokens.length === 0) {
    showError('유효한 시작 토큰을 입력하세요.');
    return;
  }
  
  const generateCount = 5; // 생성할 토큰 수
  const results = [];
  
  try {
    isGenerating = true;
    currentTokens = [...tokens];
    updateButtonStates(true);
    showLoading();
    
    for (let i = 0; i < generateCount; i++) {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `tokens=${currentTokens.join(',')}`
      });
      
      if (!response.ok) {
        throw new Error(`서버 오류: ${response.status}`);
      }
      
      const result = await response.text();
      const nextToken = extractTokenFromResult(result);
      
      if (nextToken !== null) {
        currentTokens.push(nextToken);
        results.push(nextToken);
        
        // 실시간 업데이트
        displayGenerationProgress(tokens, results);
        
        // 잠시 대기 (사용자가 과정을 볼 수 있도록)
        await new Promise(resolve => setTimeout(resolve, 300));
      }
    }
    
    displayGenerationResult(tokens, results);
    
  } catch (error) {
    console.error('시퀀스 생성 오류:', error);
    showError(`시퀀스 생성 중 오류가 발생했습니다: ${error.message}`);
  } finally {
    isGenerating = false;
    updateButtonStates(false);
    hideLoading();
  }
}

// 결과에서 토큰 추출
function extractTokenFromResult(result) {
  // HTML에서 토큰 숫자 추출
  const match = result.match(/next_token_argmax:\s*(\d+)/);
  return match ? parseInt(match[1], 10) : null;
}

// 예측 결과 표시
function displayPredictionResult(inputTokens, result) {
  const resultContainer = document.getElementById('result');
  const nextToken = extractTokenFromResult(result);
  
  resultContainer.innerHTML = `
    <div class="prediction-result">
      <div class="result-header">
        <span>🎯</span>
        <span>예측 결과</span>
      </div>
      
      <div>
        <strong>입력 토큰:</strong>
        <div class="token-sequence">
          ${inputTokens.map(token => `<span class="token">${token}</span>`).join('')}
        </div>
      </div>
            
      <div style="margin-top: 1rem; padding: 1rem; background: var(--gray-100); border-radius: var(--border-radius);">
        <small><strong>원본 응답:</strong></small>
        <pre style="margin: 0.5rem 0; white-space: pre-wrap;">${result}</pre>
      </div>
    </div>
  `;
}


// 오류 메시지 표시
function showError(message) {
  const resultContainer = document.getElementById('result');
  
  resultContainer.innerHTML = `
    <div class="prediction-result" style="border-color: var(--danger-color);">
      <div class="result-header" style="color: var(--danger-color);">
        <span>❌</span>
        <span>오류</span>
      </div>
      <p style="color: var(--danger-color); margin: 0;">${message}</p>
    </div>
  `;
}

// 결과 초기화
function clearResults() {
  const resultContainer = document.getElementById('result');
  
  resultContainer.innerHTML = `
    <div class="welcome-message">
      <h4>트랜스포머 모델 테스트</h4>
      <p>현재 모델은 교육용 데모로, 무작위 가중치를 사용합니다.</p>
    </div>
  `;
  
  document.getElementById('tokenInput').value = '';
  validateTokenInput();
}

// 설정 조정 관련 함수들
function updateConfig() {
  // 실시간으로 입력값 검증
  const dModel = parseInt(document.getElementById('dModelInput').value);
  const nHeads = parseInt(document.getElementById('nHeadsInput').value);
  
  if (dModel % nHeads !== 0) {
    document.getElementById('nHeadsInput').style.borderColor = '#dc3545';
    showTooltip(document.getElementById('nHeadsInput'), 'd_model이 n_heads로 나누어떨어져야 합니다');
  } else {
    document.getElementById('nHeadsInput').style.borderColor = '';
    hideTooltip();
  }
}

function applyConfig() {
  const config = {
    vocab_size: parseInt(document.getElementById('vocabSizeInput').value),
    d_model: parseInt(document.getElementById('dModelInput').value),
    n_layers: parseInt(document.getElementById('nLayersInput').value),
    n_heads: parseInt(document.getElementById('nHeadsInput').value),
    d_ff: parseInt(document.getElementById('dFfInput').value),
    max_seq_len: parseInt(document.getElementById('maxSeqInput').value)
  };

  // 서버에 설정 변경 요청
  fetch('/set_config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  })
  .then(res => res.json())
  .then(data => {
    if (data.success) {
      showSuccess('설정이 적용되었습니다!');
      updateConfigDisplay(config);
    } else {
      showError('설정 적용 실패: ' + (data.message || '서버 오류'));
    }
  })
  .catch(() => showError('서버와 통신할 수 없습니다.'));
}

function resetConfig() {
  const defaultConfig = {
    vocab_size: 3200,
    d_model: 128,
    n_layers: 1,
    n_heads: 2,
    d_ff: 512,
    max_seq_len: 64
  };
  
  document.getElementById('vocabSizeInput').value = defaultConfig.vocab_size;
  document.getElementById('dModelInput').value = defaultConfig.d_model;
  document.getElementById('nLayersInput').value = defaultConfig.n_layers;
  document.getElementById('nHeadsInput').value = defaultConfig.n_heads;
  document.getElementById('dFfInput').value = defaultConfig.d_ff;
  document.getElementById('maxSeqInput').value = defaultConfig.max_seq_len;
  
  updateConfigDisplay(defaultConfig);
}

function calculateParameters(config) {
  const { vocab_size, d_model, n_layers, d_ff, max_seq_len } = config;
  
  // 임베딩: tok_emb + pos_emb
  const embedding_params = vocab_size * d_model + max_seq_len * d_model;
  
  // 각 레이어: MHA (Wq,Wk,Wv,Wo) + FFN (W1,W2) + LayerNorm (2개)
  const mha_params = 4 * d_model * d_model; // Wq, Wk, Wv, Wo
  const ffn_params = d_model * d_ff + d_ff * d_model; // W1, W2
  const ln_params = 2 * d_model * 2; // ln1, ln2 각각 gamma, beta
  const layer_params = mha_params + ffn_params + ln_params;
  
  // 최종 출력: Wout + final LayerNorm
  const output_params = d_model * vocab_size + d_model * 2;
  
  return embedding_params + n_layers * layer_params + output_params;
}

function formatParams(params) {
  if (params >= 1000000) {
    return `${(params/1000000).toFixed(1)}M`;
  } else if (params >= 1000) {
    return `${(params/1000).toFixed(1)}K`;
  }
  return params.toString();
}

function showSuccess(message) {
  const result = document.getElementById('result');
  result.innerHTML = `<div class="success-message"><h4>성공</h4><p>${message}</p></div>`;
}
