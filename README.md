# mini-transformer
> 초경량 GPT/Transformer C++ 구현 및 구조 해설

- 본 문서는 Decoder-only(GPT 계열) 기반의 미니멀 Transformer를 C++로 구현한 예제와 아키텍처 설명을 제공합니다.
- C++17 참조 구현을 포함합니다.

<img width="1324" height="1686" alt="스크린샷 2025-08-18 오전 9 48 03" src="https://github.com/user-attachments/assets/c4e6065b-b945-47e5-b73c-e986bb252249" />  
<img width="1074" height="2398" alt="스크린샷 2025-08-18 오전 9 47 55" src="https://github.com/user-attachments/assets/36516c59-6e73-4161-b832-7ad73c31aa57" />  


1. [mini-transformer: 초경량 GPT/Transformer C++ 구현 및 구조 해설](#mini-transformer-초경량-gpttransformer-c-구현-및-구조-해설)
2. [Intro](#1-intro)
3. [Pipe Flow](#2-pipe-flow)
4. [구성 요소 상세](#3-구성-요소-상세)
  - [토크나이저 & 임베딩](#31-토크나이저--임베딩)
  - [트랜스포머 블록 (Pre-LN)](#32-트랜스포머-블록-pre-ln)
  - [출력 계층](#33-출력-계층)
5. [LLM과의 차이](#4-llm과의-차이)
6. [실습 스펙 제안](#5-실습-스펙-제안raspberry-pi-4-ok)
7. [C++ 구현 개요](#6-c-구현-개요)
8. [API 및 런타임 안내](#7-api-및-런타임-안내)
9. [빌드 & 실행](#8-빌드--실행)
10. [엔진 설정 및 모델 가중치 관리](#9-엔진-설정-및-모델-가중치-관리)
11. [엔진 아키텍처(상세)](#10-엔진-아키텍처상세)
12. [Front-End UI](#11-front-end-ui)
13. [라이선스/보안 스텁](#12-라이선스보안-스텁)
14. [참고](#13-참고)
15. [진행 사항 요약(변경/보강 내역)](#14-진행-사항-요약변경보강-내역)
16. [부록 A. minigpt-char.json JSON 가중치 API 문서](#부록-a-minigpt-charjson-json-가중치-api-문서)
17. [engine-config.json 설정 파일 변수 설명 및 활용 가이드](#15-engine-configjson-설정-파일-변수-설명-및-활용-가이드)
18. [부록. next_token_argmax: 수학적 정의와 트랜스포머 연결](#부록-next_token_argmax-수학적-정의와-트랜스포머-연결)
19. [모델 구성 & 추론 파이프라인 (무학습 데모)](#모델-구성--추론-파이프라인-무학습-데모)

---

## 1. Intro

| 구분 | 설명 |
| --- | --- |
| Transformer | 시퀀스 데이터를 처리하는 신경망 아키텍처. "Attention is All You Need"(2017). Self-Attention이 핵심이며 입력/출력 길이가 가변. |
| LLM (Large Language Model) | Transformer를 매우 크게 확장한 모델. 수억~수천억 파라미터. 학습 시에는 거대 미분 계산, 추론 시에는 순전파만 사용. |
| 미니멀 Transformer | LLM과 동일한 구조를 갖되, 층/차원/파라미터를 과감히 축소해 라즈베리파이 등에서도 실습 가능한 형태. |

---

## 2. Pipe Flow

Decoder-only, Pre-LN:
```
LN→MHA→Residual → LN→FFN→Residual을 n_layers 반복
```
반복 횟수: `n_layers`

> cf. GPT-2 (Radford et al., 2019)과 동일한 Pre-LN 블록 구조

#### Pipe Flow 단계별 코드/수식 대응표
| 단계      | Pipe Flow 설명                                   | 코드/수식 대응                       | 비고                       |
|-----------|--------------------------------------------------|--------------------------------------|----------------------------|
| 입력      | 입력 텍스트 → 토크나이저 → 토큰 ID                | api_server.cpp: string → tokenizer   | 텍스트를 정수 시퀀스로 변환 |
| 임베딩    | 토큰 임베딩 + 위치 임베딩 (seq_len × d_model)     | transformer.cpp: token_emb + pos_emb | 임베딩 테이블 합산         |
| 블록 반복 | n_layers 회 반복                                  | transformer.cpp: for (l=0; l<L; ++l) | 트랜스포머 스택            |
| Pre-LN    | [Pre-LN]                                         | layernorm.cpp                        | LayerNorm 먼저 적용        |
| MHA       | Multi-Head Attention                             | attention.cpp                        | 자기어텐션                 |
| Residual  | Residual Add                                     | transformer.cpp                      | 입력 + MHA 결과            |
| Pre-LN    | [Pre-LN]                                         | layernorm.cpp                        | FFN 전에도 LN              |
| FFN       | Feed Forward Network                             | ffn.cpp                              | Linear → GELU → Linear     |
| Residual  | Residual Add                                     | transformer.cpp                      | 입력 + FFN 결과            |
| 최종      | LayerNorm                                        | layernorm.cpp                        | 출력 정규화                |
| Linear    | d_model → vocab_size(=vocab)                   | transformer.cpp: out_proj(=Wout)             | 어휘 분포 투영             |
| Softmax   | 확률 분포                                        | tensor.hpp::softmax_rows                            | 확률화                     |
| 선택      | argmax / 샘플링                                  | next_token_argmax                    | 다음 토큰 결정             |

#### check point
- Self-Attention: 모든 토큰이 서로를 주목해 문맥 상호작용을 수행.
- Residual + LayerNorm: 깊이 증가에 따른 학습 불안정성 완화. 실무에서는 Pre-LN이 보편적.
- FFN: 토큰별 비선형 변환을 통해 표현력을 확장(d_ff ≈ 4 x d_model 관례).

---
## 3. 구성 요소 상세
### 3.1 토크나이저 & 임베딩

- **토크나이저**: 텍스트 → 정수 시퀀스 (예: char/BPE)
- **임베딩**:  
  입력 토큰의 임베딩과 위치 임베딩을 더해 초기 표현을 생성합니다.

  $$
  \text{Embedding}[token\_id] + \text{PosEmbedding}[position]
  $$

  위치 임베딩은 고정식(sin/cos) 또는 **학습형 테이블** 모두 가능하며, 본 구현은 학습형 테이블을 사용합니다.

**토큰 → 임베딩 공식**

- 기호:
  - \( V \): vocab\_size
  - \( D \): d\_model
  - \( T \): max\_seq\_len
  - \( E \in \mathbb{R}^{V \times D} \): 토큰 임베딩 테이블
  - \( P \in \mathbb{R}^{T \times D} \): 위치 임베딩 테이블
- 입력 시퀀스 \( x_1, x_2, ..., x_T \)에 대해:

  $$
  h^{(0)}_t = E[x_t] + P[t] \qquad (t = 1, ..., T)
  $$

- 안전 수칙: 각 token id는 \( 0 \leq id < V \)
- 임베딩 형상: `tok_emb[V, D]`, `pos_emb[T, D]`

---

### 3.2 트랜스포머 블록 (Pre-LN)

- **Pre-LN**: 각 서브레이어(MHA/FFN) **앞**에 LayerNorm, 이후 Residual Add

**Multi-Head Attention (MHA) 공식**

$$
\begin{aligned}
Q &= X W_Q \\
K &= X W_K \\
V &= X W_V \\
\text{Score} &= \frac{Q K^\top}{\sqrt{d_k}} + M \\
\text{Attention} &= \mathrm{softmax}(\text{Score}) V \\
\text{Output} &= \text{ConcatHeads} \cdot W_O
\end{aligned}
$$

- \( M \): causal mask (옵션)

**Feed Forward Network (FFN) 공식**

$$
\begin{aligned}
h_1 &= X W_1 \\
h_2 &= \mathrm{GELU}(h_1) \\
Y &= h_2 W_2 \\
\text{Output} &= X + Y \quad \text{(Residual Add)}
\end{aligned}
$$

---

### 3.3 출력 계층

- 최종 LayerNorm → Linear(\( D \to V \)) → **로짓(logits)**  
- 추론 선택(기본):  
  \[
    \text{next\_token} = \arg\max(\text{logits})
  \]
  (확률 필요시 softmax 가능)

---

## 4. LLM과의 차이

| 항목 | 미니멀 | LLM |
| --- | --- | --- |
| 파라미터 | 수만~수십만 | 수억~수천억 |
| 깊이 | 1~2층 | 24~96층 |
| d_model | 64~256 | 2048~16384 |
| vocab | 1k~5k | 32k~100k |
| 학습 | 소규모 혹은 무작위 | 거대 데이터 사전학습 |
| HW | CPU(라즈베리파이 포함) | GPU/TPU |

---

## 5. 실습 스펙 제안(Raspberry Pi 4 OK)
- n_layers=1, n_heads=2, d_model=128, seq_len=64
- vocab=2000~5000, d_ff=512
- 추론(순전파)만 구현해 토큰 단위 순환 생성

---

## 6. C++ 구현 개요

- 표준: C++17, 단일 정밀도(float)
- 최소 연산: 행렬곱(matmul), 행별 softmax, LayerNorm, MHA, FFN
- 헤더들: `include/model/tensor.hpp, layernorm.hpp, attention.hpp, ffn.hpp, transformer.hpp`
- 초기화: 정규분포(표준편차 0.02)로 파라미터 랜덤 초기화
- 제약: 학습, KV 캐시, 마스킹, 배치, 혼합정밀 등은 단순화를 위해 생략(필요시 확장 가능)

### 6.1 데이터 흐름(코드 매핑)
- 임베딩: `Transformer::embed`
- 블록 순환: `for (i: layers) Block::forward`
- 어텐션: `MultiHeadSelfAttention::forward`
- FFN: `FFN::forward`
- 출력 로짓: `matmul(X, Wout)`

---
## 7. API 및 런타임 안내

- HTTP 서버: Boost.Beast 기반의 간단한 내장 서버 제공
- 웹 데모: `/` 기본 페이지에서 입력/추론 가능
- 설정 파일: `config/engine-config.json`에서 포트, 모델 파라미터, 가중치 경로 등 지정
- 주요 CLI 옵션:
  - `--help` : 사용법 안내 출력
  - `--serve` : 서버 모드로 실행 (HTTP API 제공)
  - `--config <경로>` : 설정 파일 경로 지정
  - `--tokens "1,2,3,4"` : 입력 토큰 시퀀스로 단일 추론 데모 실행

---

## 8. 빌드 & 실행

### 의존성(Ubuntu/Debian)
```
sudo apt update
sudo apt install -y build-essential cmake git libboost-all-dev nlohmann-json3-dev
```

### 초기 구성(1회)
```
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

### 일반 빌드
```
cmake --build build -j
```

### 실행
```
./build/mini_transformer
```

### 실행 옵션(메인 바이너리)
- `--help`, `-h`: 도움말 출력
- `--config <path>`: 설정 파일 경로 지정(기본: config/engine-config.json)
- `--tokens "1,2,3,4"`: 데모용 토큰 ID CSV 입력
  - 입력은 안전 파싱되며 과도한 길이/범위는 무시되고 기본 시퀀스로 대체됩니다.
- `--serve`: 엔진을 서버 모드로 유지(Ctrl+C로 종료)
- 모델 경로 등 추가 인자를 전달하려면:  
  `RUN_ARGS="--serve --model models/minigpt-char.json" ./run_demo.sh`
  
  ### 예시 및 실행 가이드

  아래 명령어들은 mini-transformer의 주요 실행 방법과 목적, 결과를 안내합니다.

  #### 1. 실행 권한 부여 및 데모 스크립트 실행
  ```bash
  $ chmod +x run_demo.sh
  $ bash run_demo.sh
  ```
  - **목적:** `run_demo.sh` 스크립트에 실행 권한을 부여하고, 데모 실행(빌드 후 모델 추론 결과 20줄 출력).
  - **결과:** 빌드 및 실행 후, 터미널에 모델의 Top-k 결과 등 간단한 출력이 표시됩니다.

  #### 2. 도움말 출력
  ```bash
  $ ./build/mini_transformer --help
  ```
  - **목적:** 사용 가능한 명령줄 옵션과 설명을 확인.
  - **결과:** 옵션 목록과 각 기능 설명이 출력됩니다.

  #### 3. 서버 모드로 실행(HTTP API)
  ```bash
  $ ./build/mini_transformer --serve --config "$(pwd)/config/engine-config.json"
  ```
  - **목적:** HTTP 서버를 실행하여 웹 UI 및 API를 통한 추론 제공.
  - **결과:** 서버가 0.0.0.0:18080에서 실행되며, 브라우저에서 http://localhost:18080/ 접속 가능. 종료는 Ctrl+C.

  #### 4. 토큰 시퀀스 입력 데모(순전파)
  ```bash
  $ ./build/mini_transformer --config "$(pwd)/config/engine-config.json" --tokens "1,2,3,4"
  ```
  - **목적:** 지정한 토큰 ID 시퀀스(예: "1,2,3,4")로 모델 순전파 데모 실행.
  - **결과:** 입력 토큰에 대한 next_token_argmax(다음 토큰 예측) 결과가 터미널에 출력됩니다.

  ##### 추가 설명
  main.cpp내 코드에서 `간단한 미니 모델(독립 실행)`의 
  ```
    // 2) 간단한 미니 모델 데모(독립 실행) — 구조 확인용
    // 입력 토큰 CSV가 있으면 안전 파싱하여 사용, 없으면 기본 {1,2,3,4}
    const int demo_vocab = 3200;          // 데모용 vocab 상한
    const int demo_dmodel = 128;
    const int demo_nlayers = 1;
    const int demo_nheads = 2;
    const int demo_dff = 512;
    const int demo_maxseq = 64;           // 과도한 길이 방지
  ```
  #### CLI 실행 결과
  ```powershell
  mac@azabell-mac mini-transformer % ./build/mini_transformer --config "$(pwd)/config/engine-config.json" --tokens "7,7,7,13"
  [INFO] mini-transformer 시작...
  [INFO] PUBLIC KEY PATH (from config): /Users/mac/Desktop/workspace/mini-transformer/config/./public.pem
  [INFO] LICENSE FILE PATH: ./license.json
  [INFO] PUBLIC KEY PATH: /Users/mac/Desktop/workspace/mini-transformer/config/./public.pem
  [INFO] Loading weights JSON: /Users/mac/Desktop/workspace/mini-transformer/config/../models/minigpt-char.json
  [INFO] Weights populated successfully
  [INFO] API Server doc root: /Users/mac/Desktop/workspace/mini-transformer/web
  [INFO] API Server initializing at 0.0.0.0:18080
  [INFO] API Server initialized successfully.
  [INFO] Model demo forward done: logits shape (4 x 4)
  [INFO] Engine initialization complete.
  [INFO] 엔진 초기화 완료. 설정 경로: /Users/mac/Desktop/workspace/mini-transformer/config/engine-config.json
  next_token_argmax: 1020
  [INFO] mini-transformer 종료.
  ```

  #### 웹 UI에서 설정 조정 기본값을 적용한 후 [예시3] 토큰 "7,7,7,13"을 누른 결과와 next_token_argmax의 결과가 서로 동일
  ```json
  입력 토큰:
  7
  7
  7
  13
  예측된 다음 토큰:
  null
  원본 응답:
  입력 토큰: [7, 7, 7, 13]
  로짓 행렬 크기: (4 x 3200)
  Top-5 (마지막 토큰 기준)

  토큰ID	logit
  1020	0.753172
  1649	0.727642
  533	0.706434
  2807	0.67401
  1321	0.622355
  다음 토큰(argmax): 1020
  ```

  ## 실제 코드 스니펫

  ### FFN::forward (ffn.cpp)
  ```cpp
  Tensor h(x.rows(), d_ff);
  matmul(x, W1, h);
  gelu(h);                       // GeLU 근사 (tensor.hpp)
  Tensor y(x.rows(), x.cols());
  matmul(h, W2, y);
  add(y, x, out);                // Residual Add
  ```

  ### LayerNorm::forward (layernorm.cpp)
  ```cpp
  for row in x:
    mean = average(row)
    var  = variance(row)
    y = gamma * (row - mean) / sqrt(var + eps) + beta
  ```

  ### Attention::forward (attention.cpp)
  ```cpp
  // 1. Q,K,V 계산
  matmul(x, Wq, Q);
  matmul(x, Wk, K);
  matmul(x, Wv, V);

  // 2. Score = Q K^T / sqrt(d_k)
  // [옵션] causal mask 적용
  softmax_rows(score);

  // 3. Context = score V
  // 4. Head concat 후 Wo 투영
  ```

  ### Softmax/Matmul 커널 (tensor.hpp)
  ```cpp
  void softmax_rows(Tensor& m) {
    for row in m:
      float max = row.max();
      row = exp(row - max);
      row /= sum(row);
  }
  ```

  ### Transformer::next_token_argmax (transformer.cpp)
  ```cpp
  // 1) 임베딩 + pos_embed
  // 2) 블록 반복: LN → MHA → Residual → LN → FFN → Residual
  // 3) final LN, Projection
  // 4) softmax + argmax
  ```

  ---

  ## 수식 & 레퍼런스 요약 표

  | 블록                   | 수식                                                                 | 참고         |
  |------------------------|----------------------------------------------------------------------|--------------|
  | Scaled Dot-Product Attention | \(\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\)      | [1, Eq.(1)]  |
  | MHA 투영               | \(Q\!=\!XW_Q,\;K\!=\!XW_K,\;V\!=\!XW_V\); concat→\(W_O\)               | [1, §3.2.1]  |
  | FFN (코드: GeLU 근사)  | \(\mathrm{FFN}(x)=\sigma(xW_1\!+\!b_1)W_2\!+\!b_2\);<br>\(\mathrm{GeLU}(x)\!\approx\!0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))\) | [1, §3.3], [2] |
  | LayerNorm              | \(y=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta\)              | [3]          |
  | Pre-LN 잔차 배치       | \(x+\mathrm{Sublayer}(\mathrm{LN}(x))\)                               | [4]          |

  ---

  ## 참고문헌

  1. Vaswani, A. et al. *Attention Is All You Need*, NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
  2. Hendrycks, D., & Gimpel, K. *Gaussian Error Linear Units (GELUs)*, 2016. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
  3. Ba, J. L., Kiros, J. R., & Hinton, G. *Layer Normalization*, 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
  4. Xiong, R. et al. *On Layer Normalization in the Transformer Architecture*, ICML 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)

    ---

  **실행 시 참고사항**
  - `[INFO]` 로그로 설정, 가중치 로딩, 서버 초기화, 추론 결과 등이 표시됩니다.
  - `[ERROR] Failed to open weights file`은 가중치 파일 경로가 잘못되었거나 파일이 없을 때 발생합니다. config/engine-config.json의 weights_path를 확인하세요.
  - 서버 모드(`--serve`)에서는 웹 UI를 통해 직접 입력/추론이 가능합니다.
  - 데모 실행은 학습된 모델이 없을 경우 랜덤 초기화로 동작하며, 구조 확인 및 테스트 목적에 적합합니다.


### 빠른 빌드/실행 (Raspberry Pi 4, $PWD 사용)
- 현재 디렉터리를 기준으로 동작하도록 $(pwd)로 경로를 고정합니다.
```
$ bash -lc "cd $(pwd) && cmake --build build -j"
$ bash -lc "cd $(pwd) && cmake --build build -j && ./build/mini_transformer | head -n 20"
```
- 다음 과정(예시)
  - JSON 가중치를 models/ 아래에 배치하고 `config/engine-config.json`에서 `weights_type=json`, `weights_path=해당 경로`로 지정
  - 토크나이저가 BPE라면 `tokenizer/vocab.json`, `tokenizer/merges.txt` 경로를 세팅
  - 재실행하여 로딩 로그(로짓 크기 등)를 확인

### run_demo.sh(헬퍼 스크립트)
- `run_demo.sh`는 CMake 구성/빌드 후 바이너리를 실행하고 출력 20줄을 보여줍니다.
- 실행 전 `lsof`로 :$PORT 점유 프로세스를 찾아 종료(SIGTERM→SIGKILL 순)해 포트를 비운 뒤 실행합니다.
- 서버로 유지하려면: `RUN_ARGS="--serve" ./run_demo.sh`처럼 환경변수로 전달하도록 추후 확장 가능(또는 직접 바이너리를 --serve로 실행).

### 예시
```
$ chmod +x run_demo.sh
$ ./run_demo.sh
$ JOBS=4 PORT=18080 ./run_demo.sh "$(pwd)"
```

---
## 9. 엔진 설정 및 모델 가중치 관리
- 엔진의 설정 파일(`config/engine-config.json`)과 모델 가중치(JSON)의 구조, 적용 방법, 오류 대처법을 안내합니다.
- 실제 예시와 함께, 각 필드의 역할과 JSON 스키마의 형상 규칙을 설명합니다.
- 빠른 실습을 위한 샘플 설정, 가중치 파일 배치, 주요 체크리스트를 제공합니다.
- 부분 로딩, 크기 불일치, 경로 오류 등 자주 발생하는 문제와 해결법을 정리합니다.
- 실습/테스트 시 참고할 수 있는 최소 예시와 실제 적용 절차를 포함합니다.

#### 예시 설정(`config/engine-config.json`)
```
{
  "common": {
    "api_port": 18080,
    "license": "./license.json",
    "public_key_path": "./public.pem"
  },
  "tokenizer": {
    "type": "bpe",
    "vocab_path": "./tokenizer/vocab.json",
    "merges_path": "./tokenizer/merges.txt"
  },
  "model": {
    "vocab_size": 4,
    "n_layers": 1,
    "n_heads": 2,
    "d_model": 4,
    "d_ff": 4,
    "max_seq_len": 4,
    "weights_type": "json",
    "weights_path": "../models/minigpt-char.json"
  }
}
```
---

### 가중치 JSON 스키마(본 엔진 로더가 읽는 키)
```
{
  "tok_emb":  [[...], ...],               // (vocab x d_model)
  "pos_emb":  [[...], ...],               // (max_seq_len x d_model)
  "Wout":     [[...], ...],               // (d_model x vocab)
  "ln_f": { "gamma": [...], "beta": [...] },
  "blocks": [
    {
      "ln1": { "gamma": [...], "beta": [...] },
      "ln2": { "gamma": [...], "beta": [...] },
      "mha": { "Wq": [[...]], "Wk": [[...]], "Wv": [[...]], "Wo": [[...]] }, // 각 (d_model x d_model)
      "ffn": { "W1": [[...]],  "W2": [[...]] }                                // (d_model x d_ff), (d_ff x d_model)
    },
    // 반복: n_layers 개
  ]
}
```

#### 필드별 상세 설명(무엇을 의미하나)
- tok_emb (V×D): 토큰 임베딩 테이블. vocab_size(=V) 개의 토큰 각각을 d_model(=D) 차원 벡터로 매핑합니다.
- pos_emb (T×D): 위치 임베딩 테이블. 최대 시퀀스 길이 T(=max_seq_len) 위치마다 D차원 벡터를 제공합니다.
- Wout (D×V): 최종 선형 변환(프로젝션) 가중치. 블록을 통과한 D차원 표현을 어휘 V 차원 로짓으로 투영합니다.
- ln_f: 최종 LayerNorm 파라미터
  - gamma (D): 스케일 파라미터
  - beta  (D): 시프트 파라미터
- blocks (길이 L): 트랜스포머 블록별 파라미터. L은 n_layers와 같아야 합니다.
  - ln1.gamma/beta (각 D): 블록 앞단 LayerNorm 파라미터(Pre-LN)
  - ln2.gamma/beta (각 D): 어텐션 잔차 후 FFN 앞단 LayerNorm 파라미터(Pre-LN)
  - mha: 다중 헤드 자기어텐션의 선형 사상 가중치(헤드 분할은 내부에서 처리)
    - Wq, Wk, Wv, Wo (각 D×D): 쿼리/키/값/출력 투영 가중치
  - ffn: 위치별 2층 MLP(확장 후 축소)
    - W1 (D×F): 확장(입력 D → 은닉 F=d_ff)
    - W2 (F×D): 축소(은닉 F → 출력 D)

#### 형상(Shape)과 인덱싱 규칙
- JSON의 2D 행렬은 “행 배열들의 배열”입니다. 즉, 바깥 배열의 길이가 행(row), 안쪽 배열의 길이가 열(col)입니다.
- 본 엔진의 기대 형상 요약
  - tok_emb: [vocab_size, d_model]
  - pos_emb: [max_seq_len, d_model]
  - Wout:    [d_model, vocab_size]
  - Wq/Wk/Wv/Wo: [d_model, d_model]
  - W1: [d_model, d_ff], W2: [d_ff, d_model]

#### 엔진 설정값과의 매핑(반드시 일치)
- V=vocab_size, D=d_model, L=n_layers, F=d_ff, T=max_seq_len (n_heads는 내부 계산에 사용되며 가중치 행렬 형상 자체에는 직접 등장하지 않습니다)
- config/engine-config.json의 model.{vocab_size,d_model,n_layers,d_ff,max_seq_len}와 위 형상이 모두 일치해야 “완전 로딩”이 됩니다.
  - 일부만 맞으면 해당 텐서만 적용되고 나머지는 랜덤 초기화로 남습니다(부분 로딩).

#### 필수/선택
- 필수에 가까움: tok_emb, pos_emb, Wout, blocks[].{mha, ffn} 및 각 ln의 gamma/beta
  - 누락되면 해당 모듈은 랜덤 초기화로 대체 가능(추론은 동작하나 결과 품질 저하)
- 선택: blocks 길이를 n_layers보다 길게 넣어도 앞의 n_layers만 사용합니다.

#### 값의 범위/자료형
- float(실수)만 허용. NaN/Inf는 허용되지 않습니다.
- 보통 가중치는 학습에서 나온 실수이며, 정규분포 기반 초기값을 사용할 수도 있습니다.

#### 흔한 오류와 해결
- 행/열 뒤집힘: 예컨대 Wout를 [V,D]로 저장하면 크기 불일치로 무시됩니다. 반드시 [D,V]로 저장하세요.
- 길이 불일치: blocks 길이가 n_layers와 달라서 일부 블록만 적용됨 → n_layers에 맞춰 수정.
- 큰 파일 파싱 지연: 공백 제거, 소수점 자릿수 축소, 필요시 바이너리 포맷(추후 로더 확장) 고려.

---

## 10. 엔진 아키텍처(상세)
```text
데이터 경로 (mini-transformer: Decoder-only, Pre-LN)

[입력 텍스트]
     |
     v
[토크나이저: 문자열 -> 토큰 ID]        (api_server.cpp / tokenizer)
     |
     v
[토큰 임베딩 + 위치 임베딩]            (transformer.cpp: tok_emb + pos_emb)
 (shape: seq_len x d_model)
     |
     v
+---------------- Transformer Block #1 ----------------+  (n_layers 반복)
|  [LayerNorm (Pre-LN)]                               |   (layernorm.cpp)
|        |                                            |
|        v                                            |
|  [Multi-Head Self-Attention]                        |   (attention.cpp)
|    - Q,K,V = X*Wq, X*Wk, X*Wv                       |  // 입력 X에서 쿼리/키/밸류 벡터 생성 (선형 변환)
|    - Score = (Q K^T) / sqrt(d_k)                    |  // 유사도 점수 행렬 계산 (내적 기반, 차원수로 정규화)
|    - Causal mask                                    |  // 미래 토큰을 보지 못하도록 상삼각 부분 -∞ 처리
|    - softmax_rows(score)                            |  // 각 행(현재 토큰 기준) 확률 분포로 정규화
|    - Context = score * V -> concat -> Wo            |  // 가중합으로 문맥 벡터 생성 → 헤드 합치고 최종 선형 변환
|        |                                            |
|        +------ Residual Add ------------------------+   (transformer.cpp)
|                                                     |
|        [LayerNorm (Pre-LN)]                         |   (layernorm.cpp)
|                |                                    |
|                v                                    |
|        [Feed Forward Network (FFN)]                 |   (ffn.cpp)
|        - y = GeLU(x W1) W2                          |   (tensor.hpp: gelu)
|                |                                    |
|                +------ Residual Add ----------------+   (transformer.cpp)
|                                                     |
+-----------------------------------------------------+
     |
     v
[최종 LayerNorm]                                   (layernorm.cpp)
     |
     v
[Projection: d_model -> vocab_size (Wout)]          (transformer.cpp)
     |
     v
[로짓(logits)에서 마지막 토큰 T만 사용]           (z_T = u_T Wout)
     |
     v
[argmax 선택(기본) / softmax 확률화는 옵션]        (next_token_argmax)
     |
     v
[토큰 샘플링/선택 -> 반복(오토리그레시브)]

```
> `Pre-LN` : 각 서브레이어 앞에 LN, 뒤에 Residual(Add).  
`마스킹/KV 캐시` : 현재 미구현(옵션으로 확장 가능).   
`선택 로직` : 기본 argmax; 확률 필요 시 softmax_rows 활용.    
`형상` : 임베딩/프로젝션 등은 README의 JSON 스키마와 config/engine-config.json의 V/D/L/F/T에 맞춤.  

##  Causal Mask란?

언어 모델(GPT 계열)은 **오토리그레시브(Autoregressive)** 방식으로 학습합니다.  
즉, 현재 토큰을 예측할 때 **과거와 자기 자신만 참조**해야 하고, 미래 토큰은 보면 안 됩니다.  

이를 위해 어텐션 스코어 계산 시 **Causal Mask**를 적용합니다.  

---

###  수식 정의

#### 일반 어텐션 (마스크 없음)

$$
\text{Score}_{i,j} = \frac{Q_i K_j^\top}{\sqrt{d_k}}
$$

#### Causal Mask 적용

$$
\text{Score}_{i,j} =
\begin{cases}
\dfrac{Q_i K_j^\top}{\sqrt{d_k}}, & j \leq i \\\\
-\infty, & j > i
\end{cases}
$$

- \( j \leq i \): 현재 위치 \(i\)의 토큰은 과거(또는 자기 자신)까지는 볼 수 있음  
- \( j > i \): 미래 토큰은 \(-\infty\) 처리하여 softmax에서 확률 0이 되도록 함  

---

###  직관적 이해

- 마스크 없을 때: 모든 토큰이 서로를 참조 → 미래 정보까지 유출됨(치팅 발생)  
- 마스크 적용 시: 현재 위치는 **자신과 과거만 참조** → 올바른 언어 모델 학습 보장  

#### 예시 (시퀀스 길이 T=4)

$$
\text{Mask} =
\begin{bmatrix}
0 & -\infty & -\infty & -\infty \\\\
0 & 0 & -\infty & -\infty \\\\
0 & 0 & 0 & -\infty \\\\
0 & 0 & 0 & 0
\end{bmatrix}
$$

---

###  정리

- **왜 필요한가?**  
  미래 토큰을 미리 보는 "치팅"을 막기 위해.  
- **어떻게 구현되나?**  
  Softmax 직전에 스코어 행렬의 상삼각 부분을 \(-\infty\)로 채움.  
- **결과**  
  GPT 모델은 `P(x_t | x_1, ..., x_{t-1})` 분포만을 올바르게 학습/추론하게 됨.  


## 구현 세부
- 수학 커널: `include/model/tensor.hpp`에 2D 텐서, 순차 matmul, 행별 softmax 구현(CPU 단일 스레드, 캐시 친화 최적화는 최소화).
- 레이어: `layernorm.hpp`, `attention.hpp`, `ffn.hpp`에 Pre-LN, MHA, FFN 순전파 구현.
- 네트워크: `transformer.hpp`에 임베딩/포지셔널 테이블, 블록 루프, 최종 프로젝션. `init_params()`로 정규분포(σ=0.02) 초기화.
- 엔진: `include/engine/engine.hpp`, `src/engine/engine.cpp`에서 설정 로드, 라이선스 체크 스텁, 모델 구성, HTTP 서버 초기화까지 담당.
- 설정 파서: `src/utils/utils.cpp`가 `config/engine-config.json`을 읽어 토크나이저/모델 파라미터를 구성.

## 제약과 확장 포인트
- 현재 causal mask, KV 캐시는 미구현(교재용 단순화). 필요 시 `attention.hpp`에서 score 마스킹 추가로 확장.
- 가중치 로더: README 9-b의 JSON 스키마를 지원하며, d_model/n_heads/n_layers/d_ff/vocab_size/max_seq_len 일치 시 로딩합니다.
- 멀티스레딩, SIMD/BLAS 최적화는 의도적으로 제외. RPi4에서도 동작하도록 간결성 우선.

---

## 11. Front-End UI
- 최소 페이지: `web/index.html`. htmx 요청으로 간단한 추론 엔드포인트를 연동할 수 있도록 확장 가능.
- 스타일: Pico.css를 참조해 가벼운 기본 스타일 구성 권장.

---

## 12. 라이선스/보안 스텁
- `secure/secure.*`에 무결성/서명 검증용 스텁 함수가 있으며, 현재는 항상 통과하도록 구성. 실제 서비스에서는 교체 필요.

---

## 13. 참고
- Vaswani et al., 2017, Attention Is All You Need
- 공개된 소형 GPT 예제 구현들(아키텍처 비교용)
- Boost.Asio/Beast 문서

---

## 14. 진행 사항 요약(변경/보강 내역)

### 아래와 같이 진행하였다.

- .trt 모델 관련 권고와 CPU 초경량 모델 추천
  - Raspberry Pi 4에서는 TensorRT(.trt)를 실행할 수 없다. .trt는 NVIDIA GPU/Jetson에 종속되며, 보통 장치에서 직접 빌드해야 한다. Jetson을 쓴다면 NGC/TRT-LLM 예제를 따라 장치 위에서 ONNX→.trt로 변환하면 된다.
  - Pi4에서는 CPU 전용 초경량 텍스트 모델로 실습하는 편이 현실적이다.
  - 문자 단위(Char-level) GPT: tiny-shakespeare 기반, 수십만~1M 파라미터. 교육·구조 이해에 최적. 공개 체크포인트는 형식이 제각각이라 본 엔진 스키마(JSON)로 변환해 쓰는 방식을 권장.
  - 작은 BPE 미니 GPT: vocab≈3k, d_model=128, n_heads=2, n_layers=1, d_ff=512 정도(≈1~2M 파라미터). 소규모 데이터로 직접 학습 후 JSON 스키마로 내보내 사용.
- 다운로드/소스 추천
  - 데이터셋: tiny-shakespeare(문자 수준) 또는 소형 한글 코퍼스.
  - 모델 체크포인트: Hugging Face Hub에서 “char-level gpt”, “tiny shakespeare gpt”, “onnx gpt2 int8” 등의 키워드로 소형/CPU 친화 모델을 탐색. 공개 체크포인트를 쓰면 본 엔진의 JSON 스키마로 변환 필요.
  - ONNX Runtime CPU 모델도 가능하나 Pi4에서는 gpt2 계열도 꽤 느릴 수 있다. 구조 학습 목적이면 char/BPE 미니가 적합.
- 엔진 패치(.json 설정 · 가중치 로더 · README 보강)
  - `config/engine-config.json`에 토크나이저/가중치 항목을 추가: tokenizer(type|vocab_path|merges_path), model(weights_type|weights_path)
  - `src/engine/engine.cpp`에 가중치 JSON 로더 구현(스키마: tok_emb, pos_emb, Wout, ln_f, blocks[…]{ln1,ln2,mha,ffn})
  - 스키마와 d_model/n_heads/n_layers/d_ff/vocab_size/max_seq_len 일치 시 로딩
  - README.md에 트랜스포머 경로 다이어그램, .trt 관련 안내, CPU 대안 모델, 엔진 구조/스키마 설명 보강
- 코드 전반 주석을 한국어로 정비
  - include/src의 엔진/모델/유틸/보안/서버 파일에 초보자용 상세 주석 반영

### 최소 절차
- 제공된 예시 가중치 `models/minigpt-char.json`를 사용하거나, 동일 스키마(JSON)로 직렬화한 자체 가중치를 원하는 경로에 둔다.
- `config/engine-config.json`에서 `model.weights_type`을 `"json"`으로, `model.weights_path`를 실제 파일 경로(예: `"./models/minigpt-char.json"`)로 설정한다. 토크나이저가 BPE면 `tokenizer` 경로도 함께 세팅한다.
- 빌드 후 실행하면 엔진이 JSON 가중치를 읽어 모델을 구성하고, 터미널/브라우저에서 결과를 확인한다. (브라우저: http://localhost:18080/)

### 보충
- .trt가 꼭 필요하면 Jetson(Orin/Nano 등)이나 NVIDIA GPU 환경에서 해당 장치 전용으로 직접 빌드해야 한다. 사전 빌드된 .trt 배포물은 장치/드라이버 의존성 때문에 재사용이 어렵다.
- 본 레포는 구조 학습/실습 목적이라 Pi4에서도 무리 없이 동작하는 단순·가벼운 순전파 구현에 초점을 두었다. 필요 시 causal mask/KV 캐시/양자화/BLAS 최적화 등을 단계적으로 추가하면 된다.

### HTML 페이지를 유지해서 보기
- 서버 유지 실행으로 백엔드를 켭니다.
>$ mkdir build 후에는 아래 순서로 진행하세요.


```bash
# 1. CMake 구성(Release 모드)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 2. 빌드(병렬)
cmake --build build -j

# 3. 실행(서버 유지)
./build/mini_transformer --serve --config "$(pwd)/config/engine-config.json"
```

- 브라우저에서 http://localhost:18080/ 접속
- 종료는 Ctrl+C
또는 
$ ./build/mini_transformer --serve --config "$(pwd)/config/engine-config.json"
또는
$ ./run_demo.sh

### 예시 분석 모델 파일(models)
- 예시 JSON 가중치를 `models/minigpt-char.json`로 제공했습니다(소형 데모 스키마).


# 2) 빌드/실행 (서버 유지)
```
$ cmake --build build -j && ./build/mini_transformer --serve --config "$(pwd)/config/engine-config.json"
```
# 3) 브라우저 접속
> http://localhost:18080/

# 4) 빠른 확인
```
$ ./build/mini_transformer --config "$(pwd)/config/engine-config.json" --tokens "7,7,7,13"
```
---

## 부록 A. `minigpt-char.json` JSON 가중치 API 문서

이 JSON은 본 엔진의 “자연어용 Decoder-only Transformer(GPT 계열)” 순전파에 필요한 가중치를 정의합니다. 목적은 “다음 토큰 예측(next-token prediction)”이며, 토크나이저로부터 얻은 정수 토큰 시퀀스를 입력으로 받습니다.

#### 핵심 개념 요약
- 모델 형태: Decoder-only(Pre-LN) Transformer 블록 n_layers 반복
- 입력 표현: 토큰 임베딩(tok_emb) + 위치 임베딩(pos_emb)
- 블록 내부: LayerNorm → MHA → Residual → LayerNorm → FFN → Residual
- 출력: 최종 LayerNorm(ln_f) → 선형사상(Wout) → 로짓

#### 파일 일반 규칙
- 인코딩: UTF-8 JSON
- 수치형: 실수(JSON number, double로 파싱 후 float로 저장)
- NaN/Inf 불가, 유한 실수만 허용
- 큰 파일: 수 MB~수십 MB도 가능하나 Pi4 등 저사양에서는 메모리/파싱 시간이 증가하므로 주의

#### 상위 키(Top-level)
- tok_emb: float[row=V, col=D] — 토큰 임베딩(Embedding). 크기: vocab_size × d_model
- pos_emb: float[row=T, col=D] — 위치 임베딩(Position Embedding). 크기: max_seq_len × d_model
- Wout: float[row=D, col=V] — 최종 투영(Linear). 크기: d_model × vocab_size
- ln_f: 객체 — 최종 LayerNorm 파라미터
  - gamma: float[D]
  - beta:  float[D]
- blocks: 배열 길이 L — 각 트랜스포머 블록의 파라미터 모음

#### 블록 객체(blocks[i])
- ln1: 객체 — 첫 LayerNorm
  - gamma: float[D]
  - beta:  float[D]
- ln2: 객체 — 둘째 LayerNorm
  - gamma: float[D]
  - beta:  float[D]
- mha: 객체 — 다중헤드 자기어텐션(내부에서는 단일 행렬로 head를 분할해 사용)
  - Wq: float[D × D]
  - Wk: float[D × D]
  - Wv: float[D × D]
  - Wo: float[D × D]
- ffn: 객체 — 위치별 FFN
  - W1: float[D × F] (확장)
  - W2: float[F × D] (축소)

#### 차원 기호와 제약
- V = vocab_size, D = d_model, L = n_layers, H = n_heads, F = d_ff, T = max_seq_len
- 토큰/위치/출력의 축 방향은 본 엔진에서 다음과 같이 고정:
  - tok_emb: [vocab, d_model]
  - pos_emb: [max_seq_len, d_model]
  - Wout:    [d_model, vocab]
  - Wq/Wk/Wv/Wo: [d_model, d_model]
  - W1: [d_model, d_ff], W2: [d_ff, d_model]
- JSON의 행렬은 바깥 배열이 행(row), 안쪽이 열(col)

#### 별칭/키 호환성
- 현재 로더는 위 키 이름만 인식합니다(aliases 미지원): tok_emb, pos_emb, Wout, ln_f, blocks[].{ln1,ln2,mha{Wq,Wk,Wv,Wo},ffn{W1,W2}}
- 다른 이름(token_embedding, position_embedding, W_out 등)을 쓰는 경우, JSON을 사전 변환해 호환 키로 바꿔야 합니다.

#### 부분 로딩(Partial load) 동작
- 각 텐서는 개별적으로 검증·적용됩니다. 크기 또는 키가 맞지 않으면 해당 텐서는 건너뛰고, 엔진은 그 부분을 랜덤 초기화로 유지합니다.
- 하나라도 성공적으로 적용되면 “Weights populated successfully” 로그가 출력될 수 있으나, 일부가 랜덤일 수 있습니다. 크기 일치 여부를 꼭 점검하세요.

#### 검증 규칙 및 대표 오류
- 행렬/벡터 크기 불일치: 해당 키 무시(로그: missing or mismatched shapes)
- 숫자 외 타입/비배열: 무시됨
- blocks 길이 > n_layers: 앞의 n_layers개만 사용
- 누락된 키: 해당 파라미터는 랜덤 초기화(학습 전용이 아니므로 실행은 가능)

#### 예시(소형 데모)
- 예시 파일: models/minigpt-char.json(학습용이 아닌 구조 확인용, 4×4 등 극소형)
- 실제 사용 시, config/engine-config.json의 model.{vocab_size,d_model,n_layers,d_ff,max_seq_len}과 JSON 차원이 정확히 일치해야 완전 로딩됩니다.

#### 엔진 로딩 워크플로우
- config/engine-config.json 설정
  - "weights_type": "json"
  - "weights_path": "./models/your_model.json"
- 실행
  - 빌드 후 ./build/mini_transformer --serve --config "$(pwd)/config/engine-config.json"
  - 브라우저로 http://localhost:18080/ 접속

#### 문제 해결 체크리스트
- 크기 불일치로 일부만 로드되는 경우: config의 V/D/L/F/T와 JSON의 모든 텐서 차원이 일치하는지 재검토
- 값이 전부 0 또는 비정상: 학습/직렬화 과정에서 dtype/스케일이 깨지지 않았는지 확인
- 대용량 JSON 로드 지연: 숫자 포맷 단순화, 공백 제거, 필요 시 바이너리 포맷 도입 고려(추후 로더 확장 가능)

---

### 15. `engine-config.json` 설정 파일 변수 설명 및 활용 가이드
이 절에서는 `config/engine-config.json` 파일의 주요 변수를 설명합니다. 각 섹션은 `common`, `tokenizer`, `model`로 나뉘며, 해당 섹션에서 사용 가능한 필드, 타입, 예시 및 설명이 포함되어 있습니다.

### 15.1 공통 설정 (`common`)
| 필드 | 타입 | 예시 | 설명 |
| --- | --- | --- | --- |
| api_port | 정수 | `18080` | API 서버 포트 번호 |
| license | 문자열 | `"./license.json"` | 라이선스 파일 경로 |
| public_key_path | 문자열 | `"./public.pem"` | 공개 키 파일 경로 |

### 15.2 토크나이저 설정 (`tokenizer`)
| 필드 | 타입 | 예시 | 설명 |
| --- | --- | --- | --- |
| type | 문자열 | `"bpe"` | 토크나이저 유형 (예: bpe, char) |
| vocab_path | 문자열 | `"./tokenizer/vocab.json"` | 어휘 파일 경로 |
| merges_path | 문자열 | `"./tokenizer/merges.txt"` | 병합 규칙 파일 경로 (BPE의 경우) |

### 15.3 모델 설정 (`model`)
| 필드 | 타입 | 예시 | 설명 |
| --- | --- | --- | --- |
| vocab_size | 정수 | `3200` | 어휘 크기 |
| n_layers | 정수 | `1` | Transformer 블록 수 |
| n_heads | 정수 | `2` | 멀티-헤드 어텐션의 헤드 수 |
| d_model | 정수 | `128` | 임베딩 차원 및 어텐션 출력 차원 |
| d_ff | 정수 | `512` | 피드 포워드 네트워크의 은닉층 차원 |
| max_seq_len | 정수 | `64` | 최대 시퀀스 길이 |
| weights_type | 문자열 | `"random"` | 가중치 파일 유형 (예: random, json) |
| weights_path | 문자열 | `""` | 가중치 파일 경로 |

각 필드는 모델 및 토크나이저의 동작 방식에 영향을 미치며, 적절한 값으로 설정해야 합니다. 예를 들어, `vocab_size`는 사용하려는 어휘 파일에 맞게 설정해야 하며, `weights_type`이 `json`인 경우 `weights_path`에 유효한 JSON 가중치 파일 경로를 지정해야 합니다.

---

## 부록. next_token_argmax: 수학적 정의와 트랜스포머 연결
여기서는 본 엔진의 `next_token_argmax`가 어떤 수식을 따르는지, 트랜스포머 순전파와 연결하여 간단히 정리합니다.

### 주요 연산 요약 표

| 개념         | 수식/설명                                   | 코드 위치(예)                       |
|--------------|---------------------------------------------|-------------------------------------|
| 임베딩 합    | $h_t^{(0)} = E_{x_t} + P_t$                 | transformer.cpp (토큰/포지션 테이블 조회+합) |
| Pre-LN       | $\tilde{h} = \mathrm{LN}(h)$                | layernorm.cpp                       |
| 어텐션 점수  | $S = \frac{QK^\top}{\sqrt{d_k}} + M$        | attention.cpp (matmul→스케일→[옵션]마스크) |
| softmax      | 행별 softmax                                | tensor.hpp::softmax_rows()          |
| 컨텍스트     | $\alpha V$ (헤드별→concat)                  | attention.cpp                       |
| FFN          | $\mathrm{GeLU}(xW_1)W_2$                    | ffn.cpp (gelu() 근사 호출)          |
| 최종 투영    | $z_T = u_T W_{\text{out}}$                  | transformer.cpp (마지막 토큰만 사용) |
| 토큰 선택    | $\arg\max_v z_T[v]$                         | Transformer::next_token_argmax()    |

### 오토리그레시브 마스킹(옵션)

- 어텐션 수식의 $M$은 causal mask를 의미합니다.  
- attention.cpp에서 마스크를 켜면, 상삼각 영역에 큰 음수(예: $-1e9$)를 더해 미래 토큰의 softmax가 0이 되도록 처리합니다.

기호: $V$(어휘), $D$(히든), $H$(헤드), $F$(FFN 차원), $L$(블록), 입력 시퀀스 $x_1..x_T$.

1) 입력 임베딩

$$
 h^{(0)}_t = E_{x_t} + P_t\quad (t=1,\dots,T)
$$

2) 각 블록 $l=1..L$ (Pre-LN → MHA → Residual → Pre-LN → FFN → Residual)

$$
\begin{aligned}
\tilde{h}^{(l-1)}_t &= \mathrm{LN}\!\big(h^{(l-1)}_t\big),\\
q^{(a)}_t &= \tilde{h}^{(l-1)}_t W^{(a)}_Q,\quad
k^{(a)}_j = \tilde{h}^{(l-1)}_j W^{(a)}_K,\quad
v^{(a)}_j = \tilde{h}^{(l-1)}_j W^{(a)}_V,\\
\alpha^{(a)}_{t,j} &= \mathrm{softmax}_j\!\left( \frac{ q^{(a)}_t {k^{(a)}_j}^{\top} }{ \sqrt{D/H} } + M_{t,j} \right),\\
o^{(a)}_t &= \sum_j \alpha^{(a)}_{t,j} v^{(a)}_j,\quad
o_t = \mathrm{concat}_a\big(o^{(a)}_t\big) \, W_O,\\
\hat{h}_t &= h^{(l-1)}_t + o_t,\\
 b_t &= \mathrm{LN}(\hat{h}_t),\quad f_t = \mathrm{GELU}(b_t W_1) W_2,\\
 h^{(l)}_t &= \hat{h}_t + f_t.
\end{aligned}
$$

3) 최종 정규화와 로짓

$$
 u_T = \mathrm{LN}(h^{(L)}_T),\quad z_T = u_T \, W_{\mathrm{out}} \in \mathbb{R}^{V}
$$

4) 확률과 선택

$$
 p(v\mid x_{1:T}) = \mathrm{softmax}(z_T)_v = \frac{e^{z_T[v]}}{\sum_{v'=0}^{V-1} e^{z_T[v']}},\quad
 \mathrm{next\_token\_argmax}(x_{1:T}) = \arg\max_v z_T[v].
$$

-- 깃 클론을 받고 보시면 수식이 깨지지 않습니다.
<img width="854" height="641" alt="스크린샷 2025-08-18 오후 12 37 34" src="https://github.com/user-attachments/assets/c364c877-7977-42a2-9ed7-2f897d71be4e" />  
